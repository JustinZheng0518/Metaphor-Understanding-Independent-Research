# Metaphor Understanding — VUA Verb Classification

> **Experiment log (2026-11-10).** The numbers below are recorded results on the **original Gao-style formatted VUA files as read by the experiment scripts**, which contain mixed text encodings and known mojibake. These runs are retained for traceability, **not presented as clean-data or publication-ready scores**. Corrected-text replication is pending.

### Data and evaluation protocol

- Original project split: **15,516 Train / 1,724 Dev / 5,873 Test**; label 1 = metaphor, label 0 = literal.
- The Gao-style formatted CSV files are not uniformly UTF-8. A number of examples contain pre-existing garbled punctuation; some scripts additionally read them using `latin1`. This permits tokenization and training but may change model inputs.
- A prior split audit also reported repeated examples and overlap across some split identifiers; preserve original split for reproduction, but acknowledge limitations in generalization claims.
- The BERT/RoBERTa results below come from **one run per model (seed 42)**. Models were chosen by Dev F1; Test was evaluated once after selection. Do not tune on Test errors.
- Do **not** compare these results to newly cleaned data without recording a new dataset version, rechecking target-word alignment, and rerunning training and evaluation.

---

## Baseline Results

The following experiments establish a progression from simple non-contextual baselines to contextual neural models on the VUA verb-classification task.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority | 0.700 | 0.000 | 0.000 | 0.000 |
| Lexical (train only) | 0.764 | 0.665 | 0.429 | 0.522 |
| Gao 2018 Lexical | 0.765 | 0.679 | 0.407 | 0.509 |
| BiLSTM + GloVe + Target Indicator | 0.7572 | 0.6139 | 0.5128 | 0.5588 |
| BiLSTM + GloVe + ELMo + Target Indicator | 0.8027 | 0.6783 | 0.6502 | 0.6640 |
| BERT-base + Target Mean Pooling | 0.8435 | 0.7638 | 0.6922 | 0.7262 |
| RoBERTa-base + Target Mean Pooling | 0.8578 | 0.8038 | 0.6956 | 0.7458 |

### Majority Baseline

The majority baseline always predicts the most frequent class in the training data.

Because literal usages are more common than metaphorical usages in the VUA verb-classification dataset, this model predicts every example as literal.

Although the model achieves an accuracy of 0.700, its metaphor precision, recall, and F1 are all 0.000. This illustrates why accuracy alone can be misleading for an imbalanced classification task.

### Lexical Baseline (Train Only)

The lexical baseline uses only the identity of the target verb and does not use sentence context.

For each verb, the proportion of metaphorical occurrences in the training set is calculated. A verb is predicted as metaphorical when more than half of its observed training occurrences are metaphorical; otherwise, it is predicted as literal.

This version constructs the lexical statistics using only the training split.

The model achieves an F1 score of 0.522, showing that lexical identity alone contains useful information for metaphor classification. However, it cannot distinguish between literal and metaphorical uses of the same verb in different contexts.

### Gao 2018 Lexical Baseline

This baseline follows the lexical-baseline setup used in Gao et al. (2018) more closely.

Like the train-only lexical baseline, prediction is based on the historical metaphor rate of the target verb rather than on the sentence context.

The Gao-style setup uses the available training and validation data to construct the lexical statistics before evaluation on the test set.

The resulting F1 score of 0.509 provides a historical reference point for the contextual neural models reproduced later in the project.

---

## Gao et al. (2018) Reproduction

We reproduced Gao et al. (2018)-style neural metaphor classification models on the VUA verb-classification dataset.

The purpose of this stage is to study the transition from static word embeddings to contextualized word representations before moving to modern Transformer-based models.

### BiLSTM + GloVe + Target Indicator

The first neural model introduces sentence context through a bidirectional LSTM.

Each token is represented using:

- a 300-dimensional pretrained GloVe embedding;
- a 50-dimensional target indicator identifying the verb being classified.

These representations are concatenated to form a 350-dimensional input vector for each token.

The sequence is then processed by:

1. a bidirectional LSTM;
2. an attention mechanism;
3. a binary classifier predicting literal or metaphorical usage.

Unlike the lexical baselines, this model can use information from surrounding words when classifying the target verb.

Test performance:

- Accuracy: 0.7572
- Precision: 0.6139
- Recall: 0.5128
- F1: **0.5588**

### BiLSTM + GloVe + ELMo + Target Indicator

The second neural model extends the previous architecture by adding pretrained ELMo contextual representations.

Each token is represented using:

- a 300-dimensional GloVe embedding;
- a 1024-dimensional ELMo contextual embedding;
- a 50-dimensional target indicator.

These features are concatenated into a 1374-dimensional representation:

`300 + 1024 + 50 = 1374`

The same BiLSTM, attention, and classification architecture is then used.

Test performance:

- Accuracy: 0.8027
- Precision: 0.6783
- Recall: 0.6502
- F1: **0.6640**

Adding ELMo improves test F1 from **0.5588** to **0.6640**, an absolute improvement of **10.52 F1 points**.

The largest improvement is also visible in recall:

`0.5128 → 0.6502`

This suggests that contextualized representations help the model identify substantially more metaphorical usages than static GloVe representations alone.

### Reproduction Summary

The experiments show a clear progression:

`Lexical information → sentence context → contextualized representations`

The lexical baselines demonstrate that target-word identity alone contains useful information about metaphor usage.

The GloVe BiLSTM introduces sentence context and improves over the Gao lexical baseline.

Adding ELMo produces a substantially larger improvement, supporting the importance of contextualized word representations for metaphor identification.

These experiments establish the historical baseline for the Transformer-based BERT and RoBERTa results recorded below.

### Reproduction Pipeline

- `src/04_prepare_gao2018_inputs.py` — builds the vocabulary and prepares GloVe and target-indicator inputs.
- `src/05_train_gao2018_bilstm.py` — trains the BiLSTM + GloVe + Target Indicator model.
- `src/06_check_elmo_alignment.py` — verifies alignment between VUA sentences and the precomputed Gao 2018 ELMo representations.
- `src/07_train_gao2018_elmo.py` — trains the BiLSTM + GloVe + ELMo + Target Indicator model.

### Generated Results

The main experiment outputs are stored in the `results/` directory:

- `gao2018_bilstm_no_elmo_history.csv`
- `gao2018_bilstm_no_elmo_test.csv`
- `gao2018_bilstm_elmo_history.csv`
- `gao2018_bilstm_elmo_test.csv`
- `baseline_results.csv` — consolidated legacy-format VUA test results
- `vua_seen_unseen_comparison.csv` — Experiment 12.1 subgroup metrics
- `vua_unseen_error_agreement.csv` — Experiment 12.2 agreement counts

Large pretrained embeddings and model checkpoints are excluded from version control through `.gitignore`.

---

## Transformer Baselines — Experiments 10 and 11

Both models fine-tune the entire pretrained encoder, mean-pool the target verb's subword hidden states from the final layer, and apply dropout plus a 2-way linear classifier. Configuration: 256-token maximum, batch size 16, learning rate 2e-5, up to 4 epochs, seed 42, checkpoint selected by **Dev F1**. These are historical results on the original formatted dataset, **not encoding-repaired runs**.

| Model | Best epoch | Best Dev F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| BERT-base (target mean) | 3 | 0.7544 | 0.8435 | 0.7638 | 0.6922 | 0.7262 |
| RoBERTa-base (target mean) | 4 | 0.7586 | 0.8578 | 0.8038 | 0.6956 | 0.7458 |

BERT Test confusion matrix: TN=3,735, FP=377, FN=542, TP=1,219. RoBERTa Test confusion matrix: TN=3,813, FP=299, FN=536, TP=1,225. The Test set contains 5,873 examples (1,761 metaphors). Differences are **descriptive**, not significance-tested or averaged across seeds.

The local experiment scripts produced `results/bert_base_targetmean_test_metrics.json`, `results/roberta_base_targetmean_test_metrics.json`, and per-example prediction CSVs. The committed aggregate scores in `results/baseline_results.csv` are transcribed from the verified run logs; local model checkpoints and per-example files are not part of this commit.

## Seen vs Unseen Verb Analysis — Experiment 12

Seen/unseen membership is based on case-normalized, whitespace-stripped **target verb lemma** in Train. Of 5,873 Test examples, 5,546 are seen and 327 are unseen (256 unique unseen lemmas). This is an observational subgroup analysis on the original split, not an independent held-out-verb training protocol.

| Model | Group | Samples | Metaphors | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| BERT | Seen | 5,546 | 1,660 | 0.7803 | 0.6952 | 0.7353 |
| BERT | Unseen | 327 | 101 | 0.5556 | 0.6436 | 0.5963 |
| RoBERTa | Seen | 5,546 | 1,660 | 0.8141 | 0.7018 | 0.7538 |
| RoBERTa | Unseen | 327 | 101 | 0.6452 | 0.5941 | 0.6186 |

Unseen agreement (327 examples): both correct **217**, both wrong **52**, BERT-only correct **22**, RoBERTa-only correct **36**. Unseen confusion matrix: BERT TN=174, FP=52, FN=36, TP=65; RoBERTa TN=193, FP=33, FN=41, TP=60. These observations are exploratory; Test errors must not drive subsequent hyperparameter selection.

## Data Quality / Next Steps

- **Known encoding issue:** existing Gao-style VUA CSV files contain mixed encoding and mojibake (including corrupted punctuation in VUA Test). Re-downloading the same Gao data bundle does not repair the text. Do not overwrite original data or historical checkpoints.
- A separate MelBERT data package also includes some garbled VUA-Verb text; its VUA-20 material may differ in formatting, target definitions, and splits. It is **not** a drop-in replacement without row-level verification.
- Before reporting publication results: establish a versioned, corrected source with provenance; verify `sentence`, `verb_idx`, `verb`, labels and split identity; rerun BERT/RoBERTa with consistent UTF-8 handling; report corrected runs separately. Keep Gao-reproduction legacy results for reference.
- All original historical experiment CSVs are preserved. No model training, checkpoint changes, or data corrections are made by this documentation update.
