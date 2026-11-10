"""
Experiment 10.1: BERT Baseline Smoke Test

Purpose:
    Verify BERT fine-tuning on VUA verb metaphor detection.

Model:
    bert-base-uncased
    Target subword mean pooling
    Dropout + Linear binary classifier

Dataset:
    VUA Train / Dev
    Original splits, unchanged

Research question:
    Can BERT learn to classify a target verb's usage
    as literal or metaphorical using its contextual
    representation?

This is a small smoke test, NOT a final experiment.
No checkpoints or evaluation results are saved.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score


# ====================================================
# 1. Configuration
# ====================================================

MODEL_NAME = "bert-base-uncased"

TRAIN_PATH = Path("data/VUA/VUA_formatted_train.csv")
DEV_PATH = Path("data/VUA/VUA_formatted_val.csv")

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 2e-5

SMOKE_TRAIN_SIZE = 64
SMOKE_DEV_SIZE = 32


# ====================================================
# 2. Environment
# ====================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== ENVIRONMENT =====")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ====================================================
# 3. Tokenizer
# ====================================================

print("\n===== TOKENIZER =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

assert tokenizer.is_fast

print("Tokenizer loaded:", MODEL_NAME)


# ====================================================
# 4. Dataset
# ====================================================

class VUABERTDataset(Dataset):

    def __init__(self, path, limit=None):

        df = pd.read_csv(
            path,
            encoding="latin1"
        )

        if limit is not None:
            df = df.head(limit)

        self.samples = []

        for _, row in df.iterrows():

            words = str(row["sentence"]).split()
            target_idx = int(row["verb_idx"])
            label = int(row["label"])

            if not 0 <= target_idx < len(words):
                raise ValueError("Invalid target index")

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH
            )

            word_ids = encoding.word_ids()

            target_mask = [
                1 if word_id == target_idx else 0
                for word_id in word_ids
            ]

            if sum(target_mask) == 0:
                raise ValueError(
                    f"Target lost after tokenization: "
                    f"{words[target_idx]}"
                )

            # Ensure all target subwords were retained.
            full_target = tokenizer(
                words[target_idx],
                add_special_tokens=False
            )["input_ids"]

            if sum(target_mask) != len(full_target):
                raise ValueError(
                    f"Partial target truncation: "
                    f"{words[target_idx]}"
                )

            self.samples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "target_mask": target_mask,
                "label": label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        return {
            "input_ids": torch.tensor(
                sample["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                sample["attention_mask"], dtype=torch.long
            ),
            "target_mask": torch.tensor(
                sample["target_mask"], dtype=torch.float32
            ),
            "labels": torch.tensor(
                sample["label"], dtype=torch.long
            )
        }


print("\n===== LOADING DATA =====")

train_dataset = VUABERTDataset(
    TRAIN_PATH,
    limit=SMOKE_TRAIN_SIZE
)

dev_dataset = VUABERTDataset(
    DEV_PATH,
    limit=SMOKE_DEV_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Train samples:", len(train_dataset))
print("Dev samples:", len(dev_dataset))


# ====================================================
# 5. BERT Model
# ====================================================

class BERTMetaphorClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.bert.config.hidden_size

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

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [batch, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state

        # [batch, seq_len, 1]
        mask = target_mask.unsqueeze(-1)

        # Mean pooling of target subwords.
        target_sum = (
            hidden_states * mask
        ).sum(dim=1)

        target_count = mask.sum(
            dim=1
        ).clamp(min=1.0)

        target_repr = target_sum / target_count

        target_repr = self.dropout(target_repr)

        # [batch, 2]
        logits = self.classifier(target_repr)

        return logits


print("\n===== LOADING BERT =====")

model = BERTMetaphorClassifier().to(device)

print("Hidden size:", model.bert.config.hidden_size)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()
        if p.requires_grad)
)


# ====================================================
# 6. Optimizer / Loss
# ====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()


# ====================================================
# 7. Smoke Training
# ====================================================

print("\n===== SMOKE TRAINING =====")

model.train()

total_loss = 0.0

for step, batch in enumerate(train_loader, start=1):

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    optimizer.zero_grad()

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_mask=batch["target_mask"]
    )

    loss = criterion(
        logits,
        batch["labels"]
    )

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

    print(
        f"Step {step}/{len(train_loader)} | "
        f"Loss: {loss.item():.4f}"
    )

print(
    "Average train loss:",
    round(total_loss / len(train_loader), 4)
)


# ====================================================
# 8. Dev Evaluation
# ====================================================

print("\n===== DEV EVALUATION =====")

model.eval()

all_labels = []
all_predictions = []

with torch.inference_mode():

    for batch in dev_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_mask=batch["target_mask"]
        )

        predictions = logits.argmax(dim=-1)

        all_labels.extend(
            batch["labels"].cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )


f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

print("Dev F1 (SMOKE ONLY):", round(f1, 4))

if torch.cuda.is_available():

    print(
        "Peak GPU memory (GB):",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3
        )
    )

print("\n===== SMOKE TEST COMPLETE =====")
print("Forward pass: PASS")
print("Backward pass: PASS")
print("Optimizer update: PASS")
print("Dev evaluation: PASS")
print("No checkpoint or test predictions saved.")