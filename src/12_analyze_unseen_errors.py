
"""
Experiment 12.2: BERT vs RoBERTa Unseen Verb Error Analysis

Purpose:
    Analyze prediction agreement and disagreement
    between BERT and RoBERTa on unseen verb lemmas.

Models:
    10 - BERT-base
    11 - RoBERTa-base

Dataset:
    VUA original Train / Test splits.

Research questions:
    1. Which unseen examples do both models get wrong?
    2. Where does BERT outperform RoBERTa?
    3. Where does RoBERTa outperform BERT?
    4. Are errors mainly false positives or false negatives?

Outputs:
    results/unseen_bert_roberta_error_cases.csv
    results/unseen_bert_only.csv
    results/unseen_roberta_only.csv
    results/unseen_both_wrong.csv

No training, no model modifications.
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = Path("data/VUA")
RESULTS_DIR = Path("results")

TRAIN_PATH = DATA_DIR / "VUA_formatted_train.csv"
TEST_PATH = DATA_DIR / "VUA_formatted_test.csv"

BERT_PATH = (
    RESULTS_DIR / "bert_base_targetmean_test_predictions.csv"
)

ROBERTA_PATH = (
    RESULTS_DIR / "roberta_base_targetmean_test_predictions.csv"
)

SHOW_EXAMPLES = 5
CONTEXT_WINDOW = 12

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load datasets
# ============================================================

print("\n===== LOADING DATA =====")

train = pd.read_csv(
    TRAIN_PATH,
    encoding="latin1"
)

test = pd.read_csv(
    TEST_PATH,
    encoding="latin1"
)

bert = pd.read_csv(
    BERT_PATH,
    encoding="utf-8"
)

roberta = pd.read_csv(
    ROBERTA_PATH,
    encoding="utf-8"
)

print("Train samples:", len(train))
print("Test samples:", len(test))
print("BERT predictions:", len(bert))
print("RoBERTa predictions:", len(roberta))


# ============================================================
# 3. Verify prediction alignment
# ============================================================

print("\n===== ALIGNMENT CHECK =====")

identity_cols = [
    "text_idx",
    "sentence_idx",
    "verb",
    "sentence",
    "verb_idx",
    "label"
]

if not (len(test) == len(bert) == len(roberta)):
    raise ValueError("Dataset lengths do not match")

pd.testing.assert_frame_equal(
    test[identity_cols].reset_index(drop=True),
    bert[identity_cols].reset_index(drop=True),
    check_dtype=False
)

pd.testing.assert_frame_equal(
    test[identity_cols].reset_index(drop=True),
    roberta[identity_cols].reset_index(drop=True),
    check_dtype=False
)

for name, frame in [
    ("BERT", bert),
    ("RoBERTa", roberta)
]:
    if frame[["prediction", "p_metaphor"]].isna().any().any():
        raise ValueError(f"{name}: Missing predictions")

    if not frame["prediction"].isin([0, 1]).all():
        raise ValueError(f"{name}: Invalid predictions")

    if not frame["p_metaphor"].between(0, 1).all():
        raise ValueError(f"{name}: Invalid probabilities")

print("BERT / RoBERTa / Test alignment: PASS")


# ============================================================
# 4. Identify Unseen Verb Lemmas
# ============================================================

print("\n===== UNSEEN VERB CHECK =====")


def normalize_lemma(series):

    result = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    if result.isna().any() or result.eq("").any():
        raise ValueError("Invalid verb lemma")

    return result


train_lemmas = set(
    normalize_lemma(train["verb"])
)

df = test[identity_cols].copy()

df["row_id"] = range(len(df))

df["verb_lemma"] = normalize_lemma(
    df["verb"]
)

df["bert_pred"] = bert["prediction"].astype(int)
df["roberta_pred"] = roberta["prediction"].astype(int)

df["bert_prob"] = bert["p_metaphor"]
df["roberta_prob"] = roberta["p_metaphor"]

unseen = df[
    ~df["verb_lemma"].isin(train_lemmas)
].copy().reset_index(drop=True)

print("Train unique verb lemmas:", len(train_lemmas))
print("Unseen samples:", len(unseen))
print(
    "Unique unseen verb lemmas:",
    unseen["verb_lemma"].nunique()
)

print("\nUnseen label distribution:")
print(
    unseen["label"].value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 5. Verify Target Index and Surface Form
# ============================================================

print("\n===== TARGET VERIFICATION =====")


def extract_target(row):

    words = str(row["sentence"]).split()
    idx = int(row["verb_idx"])

    if not 0 <= idx < len(words):
        raise ValueError(
            f"Invalid target index at row {row['row_id']}"
        )

    return words[idx]


unseen["target_surface"] = unseen.apply(
    extract_target,
    axis=1
)

print("All target indices valid: PASS")


# ============================================================
# 6. Classify Prediction Agreement
# ============================================================

unseen["bert_correct"] = (
    unseen["bert_pred"] == unseen["label"]
)

unseen["roberta_correct"] = (
    unseen["roberta_pred"] == unseen["label"]
)


def classify_case(row):

    b = row["bert_correct"]
    r = row["roberta_correct"]

    if b and r:
        return "both_correct"

    if b and not r:
        return "bert_only"

    if not b and r:
        return "roberta_only"

    return "both_wrong"


unseen["case"] = unseen.apply(
    classify_case,
    axis=1
)

case_order = [
    "both_correct",
    "bert_only",
    "roberta_only",
    "both_wrong"
]

counts = (
    unseen["case"]
    .value_counts()
    .reindex(case_order, fill_value=0)
)

print("\n===== MODEL AGREEMENT =====")

for case, count in counts.items():

    percentage = count / len(unseen) * 100

    print(
        f"{case:16s}: "
        f"{count:3d} ({percentage:.2f}%)"
    )

assert counts.sum() == len(unseen)

print("\n===== AGREEMENT BY GOLD LABEL =====")

table = pd.crosstab(
    unseen["case"],
    unseen["label"]
)

table = table.reindex(
    index=case_order,
    columns=[0, 1],
    fill_value=0
)

print(table.to_string())


# ============================================================
# 7. Confusion Matrices and Metrics
# ============================================================

print("\n===== MODEL PERFORMANCE =====")


def evaluate_model(name, prediction_column):

    gold = unseen["label"].astype(int)
    preds = unseen[prediction_column].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        gold,
        preds,
        labels=[0, 1]
    ).ravel()

    print(f"\n--- {name} ---")

    print("Accuracy :", round(
        accuracy_score(gold, preds), 4
    ))

    print("Precision:", round(
        precision_score(gold, preds, zero_division=0), 4
    ))

    print("Recall   :", round(
        recall_score(gold, preds, zero_division=0), 4
    ))

    print("F1       :", round(
        f1_score(gold, preds, zero_division=0), 4
    ))

    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)

    assert tp + tn + fp + fn == len(unseen)


evaluate_model("BERT", "bert_pred")
evaluate_model("RoBERTa", "roberta_pred")


# ============================================================
# 8. Label Each Error Type
# ============================================================

def error_type(gold, pred):

    if gold == 1 and pred == 1:
        return "TP"

    if gold == 0 and pred == 0:
        return "TN"

    if gold == 0 and pred == 1:
        return "FP"

    return "FN"


unseen["bert_type"] = [
    error_type(g, p)
    for g, p in zip(
        unseen["label"],
        unseen["bert_pred"]
    )
]

unseen["roberta_type"] = [
    error_type(g, p)
    for g, p in zip(
        unseen["label"],
        unseen["roberta_pred"]
    )
]


# ============================================================
# 9. Short Target-Centered Context
# ============================================================

def context_preview(row, window=12):

    words = str(row["sentence"]).split()
    idx = int(row["verb_idx"])

    start = max(0, idx - window)
    end = min(len(words), idx + window + 1)

    before = words[start:idx]
    target = words[idx]
    after = words[idx + 1:end]

    parts = []

    if start > 0:
        parts.append("...")

    parts.extend(before)

    # Mark the exact target token.
    parts.append(f"[TARGET: {target}]")

    parts.extend(after)

    if end < len(words):
        parts.append("...")

    return " ".join(parts)


unseen["target_context"] = unseen.apply(
    lambda row: context_preview(
        row,
        CONTEXT_WINDOW
    ),
    axis=1
)


# ============================================================
# 10. Display Representative Examples
# ============================================================

print("\n===== ERROR CASE EXAMPLES =====")

for case in [
    "bert_only",
    "roberta_only",
    "both_wrong"
]:

    subset = unseen[
        unseen["case"] == case
    ]

    # Remove repeated target examples from the
    # printed examples only.
    # Keep all original rows in the saved results.
    examples = subset.drop_duplicates(
        subset=[
            "sentence",
            "verb_idx",
            "label"
        ]
    ).head(SHOW_EXAMPLES)

    print(f"\n========== {case.upper()} ==========")

    print("Total samples:", len(subset))
    print(
        "Unique displayed candidates:",
        len(
            subset.drop_duplicates(
                subset=["sentence", "verb_idx", "label"]
            )
        )
    )

    for _, row in examples.iterrows():

        print("\n" + "-" * 55)

        print("Row ID:", row["row_id"])
        print("Lemma:", row["verb_lemma"])
        print("Target surface:", row["target_surface"])

        print("Context:")
        print(row["target_context"])

        print("Gold label:", int(row["label"]))

        print(
            "BERT:",
            int(row["bert_pred"]),
            "|",
            row["bert_type"],
            "| P(metaphor)=",
            round(float(row["bert_prob"]), 4)
        )

        print(
            "RoBERTa:",
            int(row["roberta_pred"]),
            "|",
            row["roberta_type"],
            "| P(metaphor)=",
            round(float(row["roberta_prob"]), 4)
        )


# ============================================================
# 11. Save Full Results
# ============================================================

print("\n===== SAVING RESULTS =====")

full_path = (
    RESULTS_DIR / "unseen_bert_roberta_error_cases.csv"
)

unseen.to_csv(
    full_path,
    index=False,
    encoding="utf-8-sig"
)

print("Full unseen analysis:", full_path)

for case, filename in [
    ("bert_only", "unseen_bert_only.csv"),
    ("roberta_only", "unseen_roberta_only.csv"),
    ("both_wrong", "unseen_both_wrong.csv")
]:

    subset = unseen[
        unseen["case"] == case
    ]

    output_path = RESULTS_DIR / filename

    subset.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"{case}: {len(subset)} rows -> {output_path}"
    )


# ============================================================
# 12. Final Summary
# ============================================================

print("\n===== FINAL SUMMARY =====")

print("Total unseen samples:", len(unseen))
print("Both correct:", counts["both_correct"])
print("BERT only correct:", counts["bert_only"])
print("RoBERTa only correct:", counts["roberta_only"])
print("Both wrong:", counts["both_wrong"])

print("\nExperiment 12.2 completed.")
print("No model training performed.")
print("No data modified.")
print("Do not tune models using Test errors.")
