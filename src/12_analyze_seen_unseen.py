
"""
Experiment 12.1: Seen vs Unseen Verb Analysis

Purpose:
    Compare BERT and RoBERTa on verbs that appeared
    or did not appear in the training set.

Models:
    Experiment 10: BERT-base
    Experiment 11: RoBERTa-base

Data:
    Original VUA Train and Test splits.

Research question:
    How well do fine-tuned Transformer models
    generalize to unseen verb lemmas?

No training. No model changes.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path("data/VUA")
RESULTS_DIR = Path("results")

train = pd.read_csv(
    DATA_DIR / "VUA_formatted_train.csv",
    encoding="latin1"
)

bert = pd.read_csv(
    RESULTS_DIR / "bert_base_targetmean_test_predictions.csv",
    encoding="utf-8"
)

roberta = pd.read_csv(
    RESULTS_DIR / "roberta_base_targetmean_test_predictions.csv",
    encoding="utf-8"
)

# Both models must have predictions for the same
# examples in exactly the same order.
identity_cols = [
    "text_idx", "sentence_idx", "verb",
    "sentence", "verb_idx", "label"
]

assert len(bert) == len(roberta) == 5873

pd.testing.assert_frame_equal(
    bert[identity_cols].reset_index(drop=True),
    roberta[identity_cols].reset_index(drop=True),
    check_dtype=False,
)

print("Prediction alignment: PASS")

# Normalize verb lemmas.
def normalize_verb(series):
    if series.isna().any():
        raise ValueError("Missing verb lemma")
    return series.astype(str).str.strip().str.lower()

train_verbs = set(normalize_verb(train["verb"]))

bert["verb_lemma"] = normalize_verb(bert["verb"])
roberta["verb_lemma"] = normalize_verb(roberta["verb"])

for df in [bert, roberta]:
    df["verb_group"] = df["verb_lemma"].apply(
        lambda v: "Seen" if v in train_verbs else "Unseen"
    )

print("\n===== DATASET STATISTICS =====")

print("Train samples:", len(train))
print("Unique Train verb lemmas:", len(train_verbs))
print("Test samples:", len(bert))

print("\nTest verb groups:")
print(bert["verb_group"].value_counts().to_string())

print("\nTest metaphor counts by group:")
print(
    pd.crosstab(bert["verb_group"], bert["label"])
)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

def evaluate_group(model_name, df, group_name):

    if group_name == "All":
        subset = df
    else:
        subset = df[df["verb_group"] == group_name]

    if subset.empty:
        raise ValueError(f"Empty group: {group_name}")

    gold = subset["label"].astype(int)
    preds = subset["prediction"].astype(int)

    return {
        "model": model_name,
        "group": group_name,
        "samples": len(subset),
        "metaphors": int(gold.sum()),
        "accuracy": accuracy_score(gold, preds),
        "precision": precision_score(
            gold, preds, zero_division=0
        ),
        "recall": recall_score(
            gold, preds, zero_division=0
        ),
        "f1": f1_score(
            gold, preds, zero_division=0
        ),
    }


rows = []

for name, df in [
    ("BERT", bert),
    ("RoBERTa", roberta),
]:
    for group in ["All", "Seen", "Unseen"]:
        rows.append(evaluate_group(name, df, group))

results = pd.DataFrame(rows)

print("\n===== SEEN / UNSEEN RESULTS =====")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ----------------------------------------------------
# Compare BERT and RoBERTa
# ----------------------------------------------------

print("\n===== F1 DIFFERENCES =====")

for group in ["All", "Seen", "Unseen"]:

    b = results[
        (results["model"] == "BERT")
        & (results["group"] == group)
    ]["f1"].iloc[0]

    r = results[
        (results["model"] == "RoBERTa")
        & (results["group"] == group)
    ]["f1"].iloc[0]

    print(
        f"{group}: "
        f"RoBERTa - BERT = {r - b:+.4f}"
    )

# ----------------------------------------------------
# Save
# ----------------------------------------------------

output_path = (
    RESULTS_DIR / "vua_seen_unseen_comparison.csv"
)

results.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("Analysis complete. No training performed.")
