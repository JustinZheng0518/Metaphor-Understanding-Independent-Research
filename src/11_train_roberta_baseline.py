
"""
Experiment 11.2: RoBERTa Baseline Training

Purpose:
    Fine-tune roberta-base for VUA verb metaphor detection.

Model:
    RoBERTa -> target subword mean pooling
    -> dropout -> binary classifier.

Dataset:
    VUA 2018 verb classification.
    Original Train / Dev splits.

Research question:
    How does RoBERTa compare with BERT and Gao 2018
    on target verb metaphor detection?

Comparison:
    Experiment 10: BERT-base baseline.

Outputs:
    results/roberta_base_targetmean_history.csv
    results/roberta_base_targetmean_dev_predictions.csv
    results/roberta_base_targetmean_config.json
    checkpoints/roberta_base_targetmean_best.pt

Test set is NOT used during training.
"""

from pathlib import Path
import random
import json
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModel, AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "roberta-base"
NAME = "roberta_base_targetmean"

TRAIN_PATH = Path("data/VUA/VUA_formatted_train.csv")
DEV_PATH = Path("data/VUA/VUA_formatted_val.csv")

RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 4
PATIENCE = 2

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== RUN CONFIG =====")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Model:", MODEL_NAME)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)


# ============================================================
# 3. RoBERTa Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    add_prefix_space=True
)

assert tokenizer.is_fast

# RoBERTa uses <s> and </s> instead of
# BERT's [CLS] and [SEP].
assert tokenizer.num_special_tokens_to_add(pair=False) == 2

print("\n===== TOKENIZER =====")
print("Vocabulary size:", tokenizer.vocab_size)
print("Start token:", tokenizer.cls_token)
print("End token:", tokenizer.sep_token)


# ============================================================
# 4. Dataset
# ============================================================

class VUADataset(Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(
            path,
            encoding="latin1"
        ).reset_index(drop=True)

        if not set(self.df["label"].unique()).issubset({0, 1}):
            raise ValueError("Invalid label values")

        self.items = []
        self.centered_count = 0

        available = (
            MAX_LENGTH
            - tokenizer.num_special_tokens_to_add(pair=False)
        )

        for idx, row in self.df.iterrows():

            words = str(row["sentence"]).split()
            target_idx = int(row["verb_idx"])
            label = int(row["label"])

            if not 0 <= target_idx < len(words):
                raise ValueError(
                    f"Invalid target index: {path}, row {idx}"
                )

            # Tokenize the full sentence first.
            # We need the original word-to-subword mapping.
            enc = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=False
            )

            ids = enc["input_ids"]
            word_ids = enc.word_ids()

            positions = [
                i for i, word_id in enumerate(word_ids)
                if word_id == target_idx
            ]

            if not positions:
                raise ValueError(
                    f"Target not aligned: {path}, row {idx}"
                )

            if len(positions) > available:
                raise ValueError(
                    f"Target too long: {path}, row {idx}"
                )

            # Preserve the complete target verb.
            # This mirrors the BERT training procedure.
            if len(ids) > available:

                self.centered_count += 1

                first = positions[0]
                last = positions[-1]

                start = max(
                    0,
                    min(
                        first - (
                            available - len(positions)
                        ) // 2,
                        len(ids) - available
                    )
                )

                if last >= start + available:
                    start = last - available + 1

                ids = ids[start:start + available]
                word_ids = word_ids[start:start + available]

            # RoBERTa single sequence:
            # <s> sentence tokens </s>
            input_ids = (
                [tokenizer.cls_token_id]
                + ids
                + [tokenizer.sep_token_id]
            )

            target_mask = (
                [0]
                + [
                    int(word_id == target_idx)
                    for word_id in word_ids
                ]
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
                label,
                idx
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ============================================================
# 5. Dynamic Padding
# ============================================================

def collate_fn(batch):

    ids, masks, labels, row_ids = zip(*batch)

    input_ids = pad_sequence(
        [
            torch.tensor(x, dtype=torch.long)
            for x in ids
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    target_mask = pad_sequence(
        [
            torch.tensor(x, dtype=torch.float32)
            for x in masks
        ],
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
        "labels": torch.tensor(
            labels,
            dtype=torch.long
        ),
        "row_ids": list(row_ids)
    }


# ============================================================
# 6. Load Data
# ============================================================

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
# 7. RoBERTa Model
# ============================================================

class RoBERTaMetaphorClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            hidden_size,
            2
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        target_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [batch, seq_length, 768]
        hidden = outputs.last_hidden_state

        # [batch, seq_length, 1]
        mask = target_mask.unsqueeze(-1)

        # Target subword mean pooling
        target_sum = (
            hidden * mask
        ).sum(dim=1)

        target_count = mask.sum(
            dim=1
        ).clamp(min=1.0)

        target_repr = target_sum / target_count

        target_repr = self.dropout(target_repr)

        logits = self.classifier(target_repr)

        return logits


print("\n===== LOADING ROBERTA =====", flush=True)

model = RoBERTaMetaphorClassifier().to(DEVICE)

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# 8. Optimizer and LR Scheduler
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

total_steps = EPOCHS * len(train_loader)

warmup_steps = max(
    1,
    int(total_steps * WARMUP_RATIO)
)


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
# 9. Dev Evaluation
# ============================================================

def evaluate():

    model.eval()

    gold = []
    predictions = []
    probabilities = []
    row_ids = []

    total_loss = 0.0

    with torch.inference_mode():

        for batch in dev_loader:

            labels = batch["labels"].to(DEVICE)

            row_ids.extend(batch["row_ids"])

            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["target_mask"].to(DEVICE)
            )

            loss = criterion(logits, labels)

            total_loss += (
                loss.item() * len(labels)
            )

            preds = logits.argmax(dim=-1)

            probs = torch.softmax(
                logits,
                dim=-1
            )[:, 1]

            gold.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            gold,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    metrics = {
        "loss": total_loss / len(gold),
        "accuracy": accuracy_score(
            gold,
            predictions
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_metaphors": sum(predictions)
    }

    pred_df = dev_ds.df.iloc[row_ids].copy()

    pred_df["prediction"] = predictions
    pred_df["p_metaphor"] = probabilities

    return metrics, pred_df


# ============================================================
# 10. Save Experiment Configuration
# ============================================================

run_config = {
    "experiment": "11.2",
    "model": MODEL_NAME,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs_max": EPOCHS,
    "patience": PATIENCE,
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
    "note": (
        "Original VUA split; Test not evaluated. "
        "Known duplicate/overlap issues documented "
        "in Experiment 09."
    )
}

config_path = RESULTS_DIR / f"{NAME}_config.json"

config_path.write_text(
    json.dumps(run_config, indent=2),
    encoding="utf-8"
)


# ============================================================
# 11. Full Training
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

        loss_sum += (
            loss.item() * len(labels)
        )

        if (
            step % 100 == 0
            or step == len(train_loader)
        ):

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Train Loss {loss_sum / seen:.4f}",
                flush=True
            )

    # Dev evaluation
    metrics, pred_df = evaluate()

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

    # Select best model using Dev F1 only.
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

        pred_df.to_csv(
            RESULTS_DIR / f"{NAME}_dev_predictions.csv",
            index=False,
            encoding="utf-8"
        )

        print(
            "Saved best checkpoint!",
            flush=True
        )

    else:

        bad_epochs += 1

        if bad_epochs >= PATIENCE:

            print(
                "Early stopping.",
                flush=True
            )
            break


# ============================================================
# 12. Final Report
# ============================================================

print("\n===== TRAINING COMPLETE =====")

print("Best Dev F1:", round(best_f1, 4))
print("Checkpoint:", checkpoint_path)

print(
    "History:",
    RESULTS_DIR / f"{NAME}_history.csv"
)

print(
    "Dev predictions:",
    RESULTS_DIR / f"{NAME}_dev_predictions.csv"
)

print("Config:", config_path)
print("Test set: NOT EVALUATED")
