
"""
Experiment 10.3: BERT final Test evaluation.

Purpose:
    Evaluate the Dev-selected checkpoint once,
    without tuning on Test.

Model:
    bert-base-uncased + target-subword mean pooling
    + binary linear classifier.

Dataset:
    Original VUA Test split, unchanged.

Research question:
    How well does the selected BERT model detect
    metaphorical verbs on the held-out Test set?
"""

from pathlib import Path
import hashlib
import json

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

MODEL_NAME = "bert-base-uncased"
NAME = "bert_base_targetmean"
MAX_LENGTH = 256
BATCH_SIZE = 16

TRAIN_PATH = Path("data/VUA/VUA_formatted_train.csv")
DEV_PATH = Path("data/VUA/VUA_formatted_val.csv")
TEST_PATH = Path("data/VUA/VUA_formatted_test.csv")

CONFIG_PATH = Path("results") / f"{NAME}_config.json"
CHECKPOINT_PATH = Path("checkpoints") / f"{NAME}_best.pt"
RESULTS_DIR = Path("results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", DEVICE)

# Verify the original training data has not changed.
config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

assert config["model"] == MODEL_NAME
assert config["max_length"] == MAX_LENGTH

for path, config_key in [
    (TRAIN_PATH, "train_sha256"),
    (DEV_PATH, "dev_sha256"),
]:
    current_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if current_hash != config[config_key]:
        raise RuntimeError(
            f"Data file changed since training: {path}"
        )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True
)
assert tokenizer.is_fast


# ====================================================
# Dataset: same preprocessing as training
# ====================================================

class VUATestDataset(Dataset):

    def __init__(self, path):

        self.df = pd.read_csv(
            path, encoding="latin1"
        ).reset_index(drop=True)

        self.items = []

        special_count = (
            tokenizer.num_special_tokens_to_add(pair=False)
        )
        available = MAX_LENGTH - special_count

        for idx, row in self.df.iterrows():

            words = str(row["sentence"]).split()
            target_idx = int(row["verb_idx"])
            label = int(row["label"])

            if not 0 <= target_idx < len(words):
                raise ValueError(
                    f"Bad target index in Test row {idx}"
                )

            enc = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=False,
            )

            ids = enc["input_ids"]
            word_ids = enc.word_ids()

            positions = [
                i for i, w in enumerate(word_ids)
                if w == target_idx
            ]

            if not positions or len(positions) > available:
                raise ValueError(
                    f"Target alignment error in Test row {idx}"
                )

            # Same target-preserving truncation
            # strategy as the training script.
            if len(ids) > available:

                first, last = positions[0], positions[-1]

                start = max(
                    0,
                    min(
                        first - (
                            available - len(positions)
                        ) // 2,
                        len(ids) - available,
                    ),
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
                    f"Target truncated in Test row {idx}"
                )

            self.items.append(
                (input_ids, target_mask, label, idx)
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):

    ids, masks, labels, row_ids = zip(*batch)

    input_ids = pad_sequence(
        [
            torch.tensor(x, dtype=torch.long)
            for x in ids
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    target_mask = pad_sequence(
        [
            torch.tensor(x, dtype=torch.float32)
            for x in masks
        ],
        batch_first=True,
        padding_value=0.0,
    )

    attention_mask = (
        input_ids != tokenizer.pad_token_id
    ).long()

    return (
        input_ids,
        attention_mask,
        target_mask,
        torch.tensor(labels),
        list(row_ids),
    )


# ====================================================
# Model: identical architecture to training
# ====================================================

class BERTMetaphorClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            self.bert.config.hidden_size, 2
        )

    def forward(
        self, input_ids, attention_mask, target_mask
    ):

        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        mask = target_mask.unsqueeze(-1)

        target_repr = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return self.classifier(
            self.dropout(target_repr)
        )


# ====================================================
# Load Test set and best checkpoint
# ====================================================

print("\n===== TEST DATA =====")

test_ds = VUATestDataset(TEST_PATH)
print("Test samples:", len(test_ds))

loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

print("\n===== CHECKPOINT =====")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

assert checkpoint["model_name"] == MODEL_NAME

model = BERTMetaphorClassifier()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model = model.to(DEVICE)
model.eval()

print("Best Dev epoch:", checkpoint["best_epoch"])
print("Best Dev F1:", checkpoint["best_dev_f1"])


# ====================================================
# Predict
# ====================================================

row_ids = []
gold = []
preds = []
probs = []

with torch.inference_mode():

    for (
        input_ids,
        attention_mask,
        target_mask,
        labels,
        indices,
    ) in loader:

        logits = model(
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
            target_mask.to(DEVICE),
        )

        predictions = logits.argmax(dim=-1)

        probabilities = torch.softmax(
            logits, dim=-1
        )[:, 1]

        row_ids.extend(indices)
        gold.extend(labels.tolist())
        preds.extend(predictions.cpu().tolist())
        probs.extend(probabilities.cpu().tolist())


# ====================================================
# Calculate metrics
# ====================================================

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        gold,
        preds,
        average="binary",
        zero_division=0,
    )
)

tn, fp, fn, tp = confusion_matrix(
    gold, preds, labels=[0, 1]
).ravel()

metrics = {
    "model": MODEL_NAME,
    "selected_epoch": int(checkpoint["best_epoch"]),
    "best_dev_f1": float(checkpoint["best_dev_f1"]),
    "test_samples": len(gold),
    "accuracy": float(accuracy_score(gold, preds)),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}


# ====================================================
# Save results
# ====================================================

pred_df = test_ds.df.iloc[row_ids].copy()

pred_df["prediction"] = preds
pred_df["p_metaphor"] = probs

RESULTS_DIR.mkdir(exist_ok=True)

pred_path = (
    RESULTS_DIR / f"{NAME}_test_predictions.csv"
)
metrics_path = (
    RESULTS_DIR / f"{NAME}_test_metrics.json"
)

pred_df.to_csv(
    pred_path,
    index=False,
    encoding="utf-8",
)

metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)


# ====================================================
# Final report
# ====================================================

print("\n===== FINAL TEST RESULTS =====")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nPredictions:", pred_path)
print("Metrics:", metrics_path)

print("\nTest evaluation complete.")
print("Do not tune using this Test result.")
