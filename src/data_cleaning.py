"""Prepare sentence-and-target datasets from the repository's TroFi and MOH-X CSVs."""

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("verb", "sentence", "verb_idx", "label")
FILES = {
    "TroFi": Path("TroFi/TroFi_formatted_all3737.csv"),
    "MOH-X": Path("MOH-X/MOH-X_formatted_svo_cleaned.csv"),
}
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}

# CSV record numbers include the header. Each repair checks both word positions.
TROFI_REPAIRS = {
    293: ("drink", 1, "andrews", 7, "drank"),
    301: ("drink", 4, "drove", 18, "drank"),
    999: ("lend", 1, "example", 15, "lending"),
    1298: ("rest", 18, "crew", 34, "resting"),
    1327: ("ride", 5, "appropriated", 22, "riding"),
    1385: ("roll", 11, "from", 28, "rolling"),
    1393: ("roll", 3, "drops", 31, "rolling"),
    1527: ("strike", 3, "mcallister", 11, "struck"),
    3042: ("ride", 8, "rider", 15, "riding"),
    3046: ("ride", 4, "bringing", 22, "riding"),
    3050: ("ride", 4, "worried", 9, "riding"),
    3053: ("ride", 4, "distributor", 8, "riding"),
    3074: ("ride", 6, "experiencing", 20, "riding"),
    3075: ("ride", 8, "america", 11, "riding"),
    3352: ("stick", 6, "stories", 17, "stuck"),
    3564: ("target", 7, "monetary", 23, "targeting"),
    3698: ("touch", 3, "to", 10, "touching"),
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sentence_key(row):
    return tuple(re.findall(r"[^\W_]+", normalize(row["sentence"])))


def target_key(row):
    return normalize(row["sentence"]), normalize(row["verb"]), row["verb_idx"]


def read_dataset(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing required columns in {path}")
        rows = []
        for number, source in enumerate(reader, start=2):
            if None in source or any(not source.get(key) or not source[key].strip() for key in FIELDS):
                raise ValueError(f"Invalid CSV record {number} in {path}")
            row = {key: source[key].strip() for key in FIELDS}
            row["verb_idx"] = int(row["verb_idx"])
            tokens = row["sentence"].split()
            if row["label"] not in {"0", "1"} or not 0 <= row["verb_idx"] < len(tokens):
                raise ValueError(f"Invalid label or target index at record {number} in {path}")
            rows.append((number, row))
    return rows


def clean_dataset(name, records):
    audit = {"source_rows": len(records), "repairs": [], "removed": []}
    targets = defaultdict(list)
    repairs = TROFI_REPAIRS if name == "TroFi" else {}
    for number, row in records:
        tokens = row["sentence"].split()
        if number in repairs:
            verb, old_idx, old_word, new_idx, new_word = repairs[number]
            if (
                normalize(row["verb"]) != verb
                or row["verb_idx"] != old_idx
                or new_idx >= len(tokens)
                or tokens[old_idx].casefold() != old_word
                or tokens[new_idx].casefold() != new_word
            ):
                raise ValueError(f"TroFi record {number} does not match the audited source")
            row["verb_idx"] = new_idx
            audit["repairs"].append(
                {"record": number, "verb": verb, "old_index": old_idx, "new_index": new_idx}
            )
        if (
            name == "MOH-X"
            and normalize(row["verb"]) == "jam"
            and tokens[row["verb_idx"]].casefold() == "jampacked"
        ):
            audit["removed"].append(
                {"record": number, "reason": "ambiguous_jam_jampacked_target"}
            )
            continue
        targets[target_key(row)].append((number, row))

    if len(audit["repairs"]) != len(repairs):
        raise ValueError("Some audited TroFi repairs were absent from the source")

    cleaned = []
    for group in targets.values():
        if len({row["label"] for _, row in group}) > 1:
            audit["removed"].extend(
                {"record": number, "reason": "conflicting_label"} for number, _ in group
            )
        else:
            cleaned.append(group[0][1])
            audit["removed"].extend(
                {"record": number, "reason": "duplicate_target"} for number, _ in group[1:]
            )
    audit["cleaned_rows"] = len(cleaned)
    return cleaned, audit


def group_rows(rows, group_by):
    # Join rows transitively when they share a sentence or, in verb mode, a lemma.
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    sentence_heads = {}
    verb_heads = {}
    for index, row in enumerate(rows):
        head = sentence_heads.setdefault(sentence_key(row), index)
        parents[root(index)] = root(head)
        if group_by == "verb":
            head = verb_heads.setdefault(normalize(row["verb"]), index)
            parents[root(index)] = root(head)

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    return list(groups.values())


def summarize(rows):
    return {"rows": len(rows), "literal": sum(row["label"] == "0" for row in rows),
            "metaphor": sum(row["label"] == "1" for row in rows)}


def split_rows(rows, seed, group_by):
    groups = group_rows(rows, group_by)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    totals = summarize(rows)
    splits = {name: [] for name in SPLIT_RATIOS}
    counts = {name: Counter() for name in splits}

    def cost(name, group_counts):
        ratio = SPLIT_RATIOS[name]
        increase = 0
        for key, added in group_counts.items():
            goal = totals[key] * ratio
            before = counts[name][key] - goal
            increase += ((before + added) ** 2 - before ** 2) / max(goal, 1)
        return increase

    for group in groups:
        group_counts = summarize(group)
        destination = min(splits, key=lambda name: cost(name, group_counts))
        splits[destination].extend(group)
        counts[destination].update(group_counts)

    seen_sentences = set()
    seen_verbs = set()
    for name, split in splits.items():
        sentences = {sentence_key(row) for row in split}
        verbs = {normalize(row["verb"]) for row in split}
        if not split or seen_sentences & sentences or (group_by == "verb" and seen_verbs & verbs):
            raise ValueError(f"Empty split or group overlap in {name}")
        seen_sentences.update(sentences)
        seen_verbs.update(verbs)
    return splits


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Repository data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for generated files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by", choices=("sentence", "verb"), default="sentence")
    args = parser.parse_args()

    prepared = {}
    for name, filename in FILES.items():
        rows, audit = clean_dataset(name, read_dataset(args.data_dir / filename))
        splits = split_rows(rows, args.seed, args.group_by)
        audit["splits"] = {key: summarize(split) for key, split in splits.items()}
        prepared[name] = (rows, audit, splits)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"seed": args.seed, "group_by": args.group_by, "datasets": {}}
    for name, (rows, audit, splits) in prepared.items():
        destination = args.output_dir / name
        destination.mkdir()
        write_csv(destination / "cleaned.csv", rows)
        for split, split_data in splits.items():
            write_csv(destination / f"{split}.csv", split_data)
        report["datasets"][name] = audit
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
