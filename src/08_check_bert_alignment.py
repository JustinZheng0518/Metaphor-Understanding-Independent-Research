"""
Experiment 08: BERT Token Alignment Check on VUA

Purpose:
    Verify that VUA word-level target verb indices can be correctly
    aligned with BERT WordPiece subword tokens.

Dataset:
    VUA 2018 verb-classification train/validation/test splits.

Why this matters:
    VUA identifies the target verb at the word level using verb_idx.
    BERT tokenizes words into subwords, so one VUA word may correspond
    to one or multiple BERT tokens.

Example:
    VUA word:
        unbelievable

    Possible BERT tokens:
        un ##bel ##ievable

This script checks:
    1. Every target verb has at least one corresponding BERT token.
    2. How often target verbs are split into multiple subwords.
    3. Whether any sequence exceeds BERT's 512-token limit.
    4. Example word-to-subword alignments.

This script does NOT train BERT.
"""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "bert-base-uncased"

DATASETS = {
    "train": Path("data/VUA/VUA_formatted_train.csv"),
    "val": Path("data/VUA/VUA_formatted_val.csv"),
    "test": Path("data/VUA/VUA_formatted_test.csv"),
}

CSV_ENCODING = "latin1"


# ============================================================
# Load tokenizer
# ============================================================

print("===== Loading BERT Tokenizer =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

print("Tokenizer:", MODEL_NAME)
print("Vocabulary size:", tokenizer.vocab_size)


# ============================================================
# Alignment checker
# ============================================================

def check_split(split_name, csv_path):

    print(f"\n===== {split_name.upper()} =====")

    df = pd.read_csv(
        csv_path,
        encoding=CSV_ENCODING,
    )

    missing_targets = []
    multi_subword_targets = []
    over_512 = []

    target_subword_lengths = []

    examples = []

    for row_idx, row in df.iterrows():

        sentence = str(row["sentence"])
        words = sentence.split()

        target_idx = int(row["verb_idx"])

        if not (0 <= target_idx < len(words)):
            raise ValueError(
                f"Invalid target index at row {row_idx}: "
                f"{target_idx} / {len(words)}"
            )

        target_word = words[target_idx]

        # ----------------------------------------------------
        # Important:
        # is_split_into_words=True tells BERT that the VUA
        # sentence is already tokenized into words.
        # ----------------------------------------------------

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
        )

        word_ids = encoding.word_ids()

        # Find every BERT token belonging to the VUA target word.
        target_positions = [
            token_position
            for token_position, word_id in enumerate(word_ids)
            if word_id == target_idx
        ]

        if len(target_positions) == 0:

            missing_targets.append(
                (
                    row_idx,
                    target_word,
                    sentence,
                )
            )

            continue

        target_subword_lengths.append(
            len(target_positions)
        )

        if len(target_positions) > 1:

            bert_tokens = tokenizer.convert_ids_to_tokens(
                [
                    encoding["input_ids"][position]
                    for position in target_positions
                ]
            )

            multi_subword_targets.append(
                (
                    target_word,
                    bert_tokens,
                )
            )

        # BERT-base maximum sequence length is 512 tokens.
        if len(encoding["input_ids"]) > 512:

            over_512.append(
                (
                    row_idx,
                    len(encoding["input_ids"]),
                    sentence,
                )
            )

        # Save a few illustrative examples.
        if len(examples) < 10:

            bert_tokens = tokenizer.convert_ids_to_tokens(
                encoding["input_ids"]
            )

            target_bert_tokens = [
                bert_tokens[position]
                for position in target_positions
            ]

            examples.append(
                {
                    "target_word": target_word,
                    "verb_idx": target_idx,
                    "bert_positions": target_positions,
                    "bert_tokens": target_bert_tokens,
                    "sentence": sentence,
                }
            )

    # ========================================================
    # Summary
    # ========================================================

    print("Rows:", len(df))
    print("Missing target alignments:", len(missing_targets))
    print(
        "Multi-subword targets:",
        len(multi_subword_targets),
    )
    print(
        "Sequences > 512 BERT tokens:",
        len(over_512),
    )

    if target_subword_lengths:

        mean_subwords = (
            sum(target_subword_lengths)
            / len(target_subword_lengths)
        )

        print(
            "Mean BERT tokens per target word:",
            f"{mean_subwords:.3f}",
        )

    # ========================================================
    # Examples
    # ========================================================

    print("\nExample alignments:")

    for example in examples[:5]:

        print(
            f"\nTarget word: {example['target_word']}"
        )

        print(
            f"VUA verb_idx: {example['verb_idx']}"
        )

        print(
            f"BERT positions: "
            f"{example['bert_positions']}"
        )

        print(
            f"BERT target tokens: "
            f"{example['bert_tokens']}"
        )

        print(
            f"Sentence: {example['sentence']}"
        )

    if multi_subword_targets:

        print("\nFirst 10 multi-subword target examples:")

        for word, pieces in multi_subword_targets[:10]:

            print(
                f"{word:>20} -> {pieces}"
            )

    # ========================================================
    # Final status
    # ========================================================

    alignment_ok = (
        len(missing_targets) == 0
    )

    if alignment_ok:

        print(
            f"\n{split_name.upper()} alignment: PASS"
        )

    else:

        print(
            f"\n{split_name.upper()} alignment: FAIL"
        )

    return alignment_ok


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n===== BERT / VUA Target Alignment Check ====="
    )

    results = {}

    for split_name, path in DATASETS.items():

        results[split_name] = check_split(
            split_name,
            path,
        )

    print("\n===== Overall Summary =====")

    for split_name, passed in results.items():

        print(
            f"{split_name:>5}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if all(results.values()):

        print(
            "\nAll VUA target verbs can be aligned "
            "with BERT WordPiece tokens."
        )


if __name__ == "__main__":
    main()