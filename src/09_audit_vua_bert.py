"""
Experiment 09.2: VUA Overlap Investigation

Purpose:
    Identify duplicate examples and label conflicts.

Data:
    Existing VUA Train / Dev / Test splits.

Research question:
    Are the overlaps caused by duplicated records,
    repeated sentences, or different target verbs?

This script does not modify any data.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd

DATA_DIR = Path("data/VUA")

files = {
    "train": "VUA_formatted_train.csv",
    "dev": "VUA_formatted_val.csv",
    "test": "VUA_formatted_test.csv",
}

datasets = {}

for name, filename in files.items():

    df = pd.read_csv(
        DATA_DIR / filename,
        encoding="latin1"
    )

    # Normalize whitespace only
    df["sentence_key"] = (
        df["sentence"]
        .astype(str)
        .str.split()
        .str.join(" ")
    )

    # Same sentence and same target position
    df["target_key"] = list(zip(
        df["sentence_key"],
        df["verb_idx"].astype(int)
    ))

    datasets[name] = df


# ==================================================
# 1. Within-split duplicates
# ==================================================

print("\n===== WITHIN-SPLIT DUPLICATES =====")

for name, df in datasets.items():

    groups = df.groupby(
        ["sentence_key", "verb_idx"]
    )["label"]

    duplicate_count = (
        len(df) - groups.ngroups
    )

    conflicting = (
        groups.nunique() > 1
    ).sum()

    print(f"\n{name.upper()}")
    print("Total rows:", len(df))
    print("Repeated target records:", duplicate_count)
    print("Conflicting label groups:", conflicting)

    # Full-row duplicates including metadata
    exact = df.duplicated(
        subset=[
            "text_idx",
            "sentence_idx",
            "verb",
            "sentence",
            "verb_idx",
            "label"
        ]
    ).sum()

    print("Exact full-row duplicates:", exact)


# ==================================================
# 2. Cross-split overlaps
# ==================================================

print("\n===== CROSS-SPLIT OVERLAPS =====")

pairs = [
    ("train", "dev"),
    ("train", "test"),
    ("dev", "test"),
]

for a, b in pairs:

    df_a = datasets[a]
    df_b = datasets[b]

    keys_a = set(df_a["target_key"])
    keys_b = set(df_b["target_key"])

    shared_targets = keys_a & keys_b

    sentences_a = set(df_a["sentence_key"])
    sentences_b = set(df_b["sentence_key"])

    shared_sentences = sentences_a & sentences_b

    rows_a = df_a[
        df_a["target_key"].isin(shared_targets)
    ]

    rows_b = df_b[
        df_b["target_key"].isin(shared_targets)
    ]

    print(f"\n===== {a.upper()} vs {b.upper()} =====")

    print("Shared unique target examples:",
          len(shared_targets))

    print("Affected rows in", a, ":", len(rows_a))
    print("Affected rows in", b, ":", len(rows_b))

    print("Shared unique sentences:",
          len(shared_sentences))

    label_map_a = defaultdict(set)
    label_map_b = defaultdict(set)

    for _, row in rows_a.iterrows():
        label_map_a[row["target_key"]].add(
            int(row["label"])
        )

    for _, row in rows_b.iterrows():
        label_map_b[row["target_key"]].add(
            int(row["label"])
        )

    conflicts = [
        key for key in shared_targets
        if label_map_a[key] != label_map_b[key]
    ]

    print("Conflicting label groups:",
          len(conflicts))

    print("\nOverlap examples:")

    for key in list(sorted(shared_targets))[:5]:

        sentence, idx = key

        sample_a = df_a[
            df_a["target_key"] == key
        ].iloc[0]

        sample_b = df_b[
            df_b["target_key"] == key
        ].iloc[0]

        print("-" * 50)
        print("Sentence:", sentence[:150])
        print("Target index:", idx)
        print(
            f"{a} verb/label:",
            sample_a["verb"],
            sample_a["label"]
        )
        print(
            f"{b} verb/label:",
            sample_b["verb"],
            sample_b["label"]
        )
        print(
            "Text IDs:",
            sample_a["text_idx"],
            sample_b["text_idx"]
        )

print("\nInvestigation finished.")
print("Original data files were not changed.")