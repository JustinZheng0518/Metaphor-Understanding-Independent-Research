
"""
Experiment 11.3: RoBERTa Final Test Evaluation

Purpose:
    Evaluate the best Dev-selected RoBERTa checkpoint.

Model:
    roberta-base + target subword mean pooling
    + dropout + binary classifier.

Dataset:
    VUA original Test split (5873 examples).

Research question:
    How well does RoBERTa generalize to VUA Test?

Outputs:
    results/roberta_base_targetmean_test_predictions.csv
    results/roberta_base_targetmean_test_metrics.json

No training or hyperparameter tuning.
"""

from pathlib import Path
import json
import hashlib

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModel, AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ====================================================
# 1. Configuration
# ====================================================

MODEL_NAME = "roberta-base"
NAME = "roberta_base_targetmean"

MAX_LENGTH = 256
BATCH_SIZE = 16

TRAIN_PATH = Path("data/VUA/VUA_formatted_train.csv")
DEV_PATH = Path("data/VUA/VUA_formatted_val.csv")
TEST_PATH = Path("data/VUA/VUA_formatted_test.csv")

RESULTS_DIR = Path("results")
CHECKPOINT_PATH = Path(
    "checkpoints/roberta_base_targetmean_best.pt"
)

CONFIG_PATH = RESULTS_DIR / f"{NAME}_config.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== ENVIRONMENT =====")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ====================================================
# 2. Verify Training Configuration
# ====================================================

print("\n===== CONFIG CHECK =====")

config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

assert config["model"] == MODEL_NAME
assert config["max_length"] == MAX_LENGTH
assert config["pooling"] == "target_subword_mean"

for path, key in [
    (TRAIN_PATH, "train_sha256"),
    (DEV_PATH, "dev_sha256")
]:

    current_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if current_hash != config[key]:
        raise RuntimeError(
            f"Data changed since training: {path}"
        )

print("Training configuration verified.")
print("Train/Dev files unchanged.")


# ====================================================
# 3. Tokenizer
# ====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    add_prefix_space=True
)

assert tokenizer.is_fast

print("\n===== TOKENIZER =====")
print("Tokenizer:", MODEL_NAME)
print("Vocabulary size:", tokenizer.vocab_size)


# ====================================================
# 4. Test Dataset
# ====================================================

class VUATestDataset(Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(
            path,
            encoding="latin1"
        ).reset_index(drop=True)

        self.items = []
        self.centered_count = 0

        available = (
            MAX_LENGTH
            - tokenizer.num_special_tokens_to_add(
                pair=False
            )
        )

        for idx, row in self.df.iterrows():

            words = str(row["sentence"]).split()
            target_idx = int(row["verb_idx"])
            label = int(row["label"])

            if not 0 <= target_idx < len(words):
                raise ValueError(
                    f"Invalid target index: row {idx}"
                )

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

            if not positions or len(positions) > available:
                raise ValueError(
                    f"Target alignment error: row {idx}"
                )

            # Identical target-preserving truncation
            # to the training script.
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

            # RoBERTa: <s> tokens </s>
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
                    f"Target truncated: row {idx}"
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


# ====================================================
# 5. Dynamic Padding
# ====================================================

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


print("\n===== TEST DATA =====")

test_ds = VUATestDataset(TEST_PATH)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print("Test samples:", len(test_ds))
print("Target-centered examples:", test_ds.centered_count)


# ====================================================
# 6. Model - Same Architecture as Training
# ====================================================

class RoBERTaMetaphorClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            hidden_size, 2
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

        hidden = outputs.last_hidden_state
        mask = target_mask.unsqueeze(-1)

        target_sum = (
            hidden * mask
        ).sum(dim=1)

        target_count = mask.sum(
            dim=1
        ).clamp(min=1.0)

        target_repr = target_sum / target_count

        return self.classifier(
            self.dropout(target_repr)
        )


# ====================================================
# 7. Load Best Checkpoint
# ====================================================

print("\n===== CHECKPOINT =====")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

assert checkpoint["model_name"] == MODEL_NAME

model = RoBERTaMetaphorClassifier()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("Best Dev epoch:", checkpoint["best_epoch"])
print(
    "Best Dev F1:",
    round(checkpoint["best_dev_f1"], 4)
)


# ====================================================
# 8. Test Predictions
# ====================================================

print("\n===== TEST EVALUATION =====")

gold = []
predictions = []
probabilities = []
row_ids = []

with torch.inference_mode():

    for batch in test_loader:

        row_ids.extend(batch["row_ids"])

        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["target_mask"].to(DEVICE)
        )

        preds = logits.argmax(dim=-1)

        probs = torch.softmax(
            logits,
            dim=-1
        )[:, 1]

        gold.extend(
            batch["labels"].tolist()
        )

        predictions.extend(
            preds.cpu().tolist()
        )

        probabilities.extend(
            probs.cpu().tolist()
        )


# ====================================================
# 9. Metrics
# ====================================================

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        gold,
        predictions,
        average="binary",
        zero_division=0
    )
)

tn, fp, fn, tp = confusion_matrix(
    gold,
    predictions,
    labels=[0, 1]
).ravel()

metrics = {
    "model": MODEL_NAME,
    "selected_epoch": int(
        checkpoint["best_epoch"]
    ),
    "best_dev_f1": float(
        checkpoint["best_dev_f1"]
    ),
    "test_samples": len(gold),
    "accuracy": float(
        accuracy_score(gold, predictions)
    ),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}


# ====================================================
# 10. Save Results
# ====================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

prediction_df = test_ds.df.iloc[row_ids].copy()

prediction_df["prediction"] = predictions
prediction_df["p_metaphor"] = probabilities

prediction_path = (
    RESULTS_DIR / f"{NAME}_test_predictions.csv"
)

metrics_path = (
    RESULTS_DIR / f"{NAME}_test_metrics.json"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8"
)

metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)


# ====================================================
# 11. Final Report
# ====================================================

print("\n===== FINAL TEST RESULTS =====")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nPredictions:", prediction_path)
print("Metrics:", metrics_path)

print("\nTest evaluation complete.")
print("No training performed.")
print("Do not tune based on Test results.")
