
"""
Experiment 11.1: RoBERTa Baseline Smoke Test

Purpose:
    Verify RoBERTa fine-tuning for VUA verb metaphor detection.

Model:
    roberta-base
    Target-subword mean pooling
    Dropout + Linear classifier

Dataset:
    Original VUA Train / Dev splits.
    First 64 training and 32 dev samples only.

Research question:
    Does the RoBERTa training pipeline work correctly?

No checkpoints saved. No Test evaluation.
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

# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 2e-5
SEED = 42

TRAIN_PATH = Path("data/VUA/VUA_formatted_train.csv")
DEV_PATH = Path("data/VUA/VUA_formatted_val.csv")

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

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()

# ==================================================
# Tokenizer
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    add_prefix_space=True
)

assert tokenizer.is_fast

print("\n===== TOKENIZER =====")
print("Model:", MODEL_NAME)
print("Vocabulary size:", tokenizer.vocab_size)

# ==================================================
# Dataset
# ==================================================

class VUADataset(Dataset):

    def __init__(self, path, limit):

        df = pd.read_csv(
            path,
            encoding="latin1"
        ).head(limit)

        self.items = []

        for i, row in df.iterrows():

            words = str(row["sentence"]).split()
            target_idx = int(row["verb_idx"])
            label = int(row["label"])

            if not 0 <= target_idx < len(words):
                raise ValueError(f"Invalid target: {i}")

            # Full tokenization to verify truncation.
            full = tokenizer(
                words,
                is_split_into_words=True,
                truncation=False
            )

            full_positions = [
                j for j, w in enumerate(full.word_ids())
                if w == target_idx
            ]

            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length"
            )

            positions = [
                j for j, w in enumerate(enc.word_ids())
                if w == target_idx
            ]

            if not positions or len(positions) != len(full_positions):
                raise ValueError(
                    f"Target lost or truncated in row {i}"
                )

            target_mask = [
                int(w == target_idx)
                for w in enc.word_ids()
            ]

            self.items.append({
                "input_ids": torch.tensor(
                    enc["input_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    enc["attention_mask"], dtype=torch.long
                ),
                "target_mask": torch.tensor(
                    target_mask, dtype=torch.float32
                ),
                "labels": torch.tensor(
                    label, dtype=torch.long
                )
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


train_ds = VUADataset(TRAIN_PATH, 64)
dev_ds = VUADataset(DEV_PATH, 32)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)

dev_loader = DataLoader(
    dev_ds, batch_size=BATCH_SIZE, shuffle=False
)

print("\n===== DATA =====")
print("Train:", len(train_ds))
print("Dev:", len(dev_ds))

# ==================================================
# RoBERTa Model
# ==================================================

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
        self, input_ids, attention_mask, target_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state
        mask = target_mask.unsqueeze(-1)

        target_repr = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return self.classifier(
            self.dropout(target_repr)
        )


print("\n===== LOADING ROBERTA =====")

model = RoBERTaMetaphorClassifier().to(device)

print(
    "Trainable parameters:",
    sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()

# ==================================================
# Training
# ==================================================

print("\n===== SMOKE TRAINING =====")

model.train()

losses = []

for step, batch in enumerate(train_loader, 1):

    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    optimizer.zero_grad()

    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["target_mask"]
    )

    loss = criterion(logits, batch["labels"])

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    optimizer.step()
    losses.append(loss.item())

    print(
        f"Step {step}/{len(train_loader)} "
        f"Loss={loss.item():.4f}",
        flush=True
    )

print("Average loss:", round(np.mean(losses), 4))

# ==================================================
# Dev Evaluation
# ==================================================

print("\n===== DEV EVALUATION =====")

model.eval()

gold = []
preds = []

with torch.inference_mode():

    for batch in dev_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["target_mask"]
        )

        gold.extend(batch["labels"].cpu().tolist())
        preds.extend(
            logits.argmax(dim=-1).cpu().tolist()
        )

print(
    "Smoke Dev F1:",
    round(f1_score(gold, preds, zero_division=0), 4)
)

print("Predicted metaphors:", sum(preds))
print("Gold metaphors:", sum(gold))

if device.type == "cuda":
    print(
        "Peak GPU memory (GB):",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3
        )
    )

print("\n===== SMOKE TEST COMPLETE =====")
print("Training: PASS")
print("Evaluation: PASS")
print("No Test data used.")
