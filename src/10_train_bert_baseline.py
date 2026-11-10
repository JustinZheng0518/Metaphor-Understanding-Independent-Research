
"""
Experiment 10.2: BERT Baseline Training (Train + Dev ONLY)

Purpose:
    Fine-tune bert-base-uncased for VUA verb metaphor detection.

Model:
    BERT -> target subword mean pooling -> dropout -> classifier.

Dataset:
    Original VUA Train / Dev splits, unchanged.

Research question:
    How well does fine-tuned BERT identify metaphorical verbs?

Outputs:
    Best checkpoint, training history, Dev predictions, config.

Important:
    No Test evaluation in this script.
"""

from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
import transformers
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 1. Configuration
# ============================================================

TRAIN_PATH = Path("data/VUA/VUA_formatted_train.csv")
DEV_PATH = Path("data/VUA/VUA_formatted_val.csv")

RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")

NAME = "bert_base_targetmean"
MODEL_NAME = "bert-base-uncased"

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 4
PATIENCE = 2

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CKPT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# 2. Reproducibility and device
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("===== RUN CONFIG =====")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Model:", MODEL_NAME)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)


# ============================================================
# 3. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

assert tokenizer.is_fast


# ============================================================
# 4. Dataset
# ============================================================

class VUADataset(Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(
            path, encoding="latin1"
        ).reset_index(drop=True)

        if not set(self.df["label"].unique()).issubset({0, 1}):
            raise ValueError("Invalid label values")

        self.items = []
        self.centered_count = 0

        special_count = tokenizer.num_special_tokens_to_add(
            pair=False
        )

        available = MAX_LENGTH - special_count

        for idx, row in self.df.iterrows():

            words = str(row["sentence"]).split()
            target_idx = int(row["verb_idx"])

            if not 0 <= target_idx < len(words):
                raise ValueError(
                    f"Invalid target index: {path}, row {idx}"
                )

            # Tokenize without truncation first.
            enc = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=False
            )

            ids = enc["input_ids"]
            word_ids = enc.word_ids()

            positions = [
                j for j, w in enumerate(word_ids)
                if w == target_idx
            ]

            if not positions:
                raise ValueError(
                    f"Unaligned target: {path}, row {idx}"
                )

            if len(positions) > available:
                raise ValueError("Target too long")

            # Preserve target subwords when truncation is needed.
            if len(ids) > available:

                self.centered_count += 1

                first = positions[0]
                last = positions[-1]

                start = max(
                    0,
                    min(
                        first - (available - len(positions)) // 2,
                        len(ids) - available
                    )
                )

                if last >= start + available:
                    start = last - available + 1

                ids = ids[start:start + available]
                word_ids = word_ids[start:start + available]


            input_ids = (
                [tokenizer.cls_token_id]
                + ids
                + [tokenizer.sep_token_id]
            )

            # BERT single sequence: [CLS] tokens [SEP]
            target_mask = (
                [0]
                + [int(w == target_idx) for w in word_ids]
                + [0]
            )

            if (
                len(input_ids) != len(target_mask)
                or sum(target_mask) != len(positions)
            ):
                raise ValueError(
                    f"Target truncation: {path}, row {idx}"
                )

            self.items.append((
                input_ids,
                target_mask,
                int(row["label"]),
                idx
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):

    ids, masks, labels, row_ids = zip(*batch)

    input_ids = pad_sequence(
        [torch.tensor(x, dtype=torch.long) for x in ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    target_mask = pad_sequence(
        [torch.tensor(x, dtype=torch.float32) for x in masks],
        batch_first=True,
        padding_value=0.0
    )

    attention_mask = (
        input_ids != tokenizer.pad_token_id
    ).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "row_ids": list(row_ids)
    }


print("\n===== DATA PREPARATION =====", flush=True)

train_ds = VUADataset(TRAIN_PATH)
dev_ds = VUADataset(DEV_PATH)

print("Train:", len(train_ds))
print("Dev:", len(dev_ds))
print("Target-centered train:", train_ds.centered_count)
print("Target-centered dev:", dev_ds.centered_count)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

dev_loader = DataLoader(
    dev_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)


# ============================================================
# 5. Model
# ============================================================

class BERTMetaphorClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        target_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        mask = target_mask.unsqueeze(-1)

        target_sum = (hidden * mask).sum(dim=1)
        target_count = mask.sum(dim=1).clamp(min=1)

        target_repr = target_sum / target_count

        return self.classifier(
            self.dropout(target_repr)
        )


print("\n===== LOADING BERT =====", flush=True)

model = BERTMetaphorClassifier().to(DEVICE)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()
        if p.requires_grad)
)


# ============================================================
# 6. Optimizer and scheduler
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

total_steps = EPOCHS * len(train_loader)
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))


def lr_factor(step):

    if step < warmup_steps:
        return float(step + 1) / warmup_steps

    return max(
        0.0,
        (total_steps - step)
        / max(1, total_steps - warmup_steps)
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_factor
)


# ============================================================
# 7. Dev evaluation
# ============================================================

def evaluate():

    model.eval()

    gold = []
    preds = []
    probs = []
    row_ids = []

    total_loss = 0.0

    with torch.inference_mode():

        for batch in dev_loader:

            row_ids.extend(batch["row_ids"])

            labels = batch["labels"].to(DEVICE)

            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["target_mask"].to(DEVICE)
            )

            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)

            gold.extend(labels.cpu().tolist())
            preds.extend(
                logits.argmax(dim=-1).cpu().tolist()
            )
            probs.extend(
                torch.softmax(logits, dim=-1)[:, 1]
                .cpu().tolist()
            )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            gold,
            preds,
            average="binary",
            zero_division=0
        )
    )

    metrics = {
        "loss": total_loss / len(gold),
        "accuracy": accuracy_score(gold, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_metaphors": sum(preds)
    }

    pred_df = dev_ds.df.iloc[row_ids].copy()

    pred_df["prediction"] = preds
    pred_df["p_metaphor"] = probs

    return metrics, pred_df


# ============================================================
# 8. Save experiment config
# ============================================================

run_config = {
    "experiment": "10.2",
    "model": MODEL_NAME,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "pooling": "target_subword_mean",
    "train_file": str(TRAIN_PATH),
    "dev_file": str(DEV_PATH),
    "train_sha256": hashlib.sha256(
        TRAIN_PATH.read_bytes()
    ).hexdigest(),
    "dev_sha256": hashlib.sha256(
        DEV_PATH.read_bytes()
    ).hexdigest(),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "note": "Original VUA split. Test not evaluated."
}

config_path = RESULTS_DIR / f"{NAME}_config.json"

config_path.write_text(
    json.dumps(run_config, indent=2),
    encoding="utf-8"
)


# ============================================================
# 9. Full training
# ============================================================

best_f1 = -1.0
bad_epochs = 0
history = []

checkpoint_path = CKPT_DIR / f"{NAME}_best.pt"

print("\n===== FULL TRAINING =====", flush=True)

for epoch in range(1, EPOCHS + 1):

    model.train()

    loss_sum = 0.0
    seen = 0

    for step, batch in enumerate(train_loader, 1):

        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["target_mask"].to(DEVICE)
        )

        loss = criterion(logits, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        seen += len(labels)
        loss_sum += loss.item() * len(labels)

        if step % 100 == 0 or step == len(train_loader):

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Train Loss {loss_sum / seen:.4f}",
                flush=True
            )

    # Evaluate on Dev, never Test.
    metrics, predictions = evaluate()

    record = {
        "epoch": epoch,
        "train_loss": loss_sum / seen,
        **{
            f"dev_{k}": float(v)
            for k, v in metrics.items()
        }
    }

    history.append(record)

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / f"{NAME}_history.csv",
        index=False
    )

    print(
        f"\nEpoch {epoch} | "
        f"Dev Loss={metrics['loss']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f}",
        flush=True
    )

    if metrics["f1"] > best_f1:

        best_f1 = float(metrics["f1"])
        bad_epochs = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": epoch,
                "best_dev_f1": best_f1,
                "model_name": MODEL_NAME
            },
            checkpoint_path
        )

        predictions.to_csv(
            RESULTS_DIR / f"{NAME}_dev_predictions.csv",
            index=False,
            encoding="utf-8"
        )

        print("Saved best checkpoint!", flush=True)

    else:

        bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.", flush=True)
            break


# ============================================================
# 10. Summary
# ============================================================

print("\n===== TRAINING COMPLETE =====")
print("Best Dev F1:", round(best_f1, 4))
print("Checkpoint:", checkpoint_path)
print("History:", RESULTS_DIR / f"{NAME}_history.csv")
print("Dev predictions:",
      RESULTS_DIR / f"{NAME}_dev_predictions.csv")
print("Config:", config_path)
print("Test set: NOT EVALUATED")
