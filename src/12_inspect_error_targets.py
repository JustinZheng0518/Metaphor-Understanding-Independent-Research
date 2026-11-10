"""
Experiment 12.3: Inspect Unseen Error Targets

Purpose:
    Manually inspect target word positions and
    prediction disagreements.

No training or data modification.
"""

import pandas as pd

df = pd.read_csv(
    "results/unseen_bert_roberta_error_cases.csv",
    encoding="utf-8"
)

for case in [
    "bert_only",
    "roberta_only",
    "both_wrong"
]:

    subset = df[df["case"] == case]

    print(f"\n===== {case.upper()} =====")
    print("Total:", len(subset))

    for _, row in subset.head(5).iterrows():

        words = str(row["sentence"]).split()
        idx = int(row["verb_idx"])

        surface = (
            words[idx]
            if 0 <= idx < len(words)
            else "[INVALID INDEX]"
        )

        print("\n-------------------------")
        print("Lemma:", row["verb"])
        print("Target surface:", surface)
        print("Target index:", idx)
        print("Gold label:", row["label"])
        print("BERT prediction:", row["bert_pred"])
        print("RoBERTa prediction:", row["roberta_pred"])
        print("Sentence:", str(row["sentence"])[:180])

print("\nInspection complete.")
