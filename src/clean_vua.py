import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from data_cleaning import (
    FILES,
    clean_dataset,
    read_dataset,
    sentence_key,
    split_rows,
    summarize,
    target_key,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "metaphor_cleaned"
VUA = ROOT / "data_new" / "VUA"
TRAIN_GOLD = ROOT / "naacl_flp_train_gold_labels" / "verb_tokens.csv"
SEED = 42


def output_exists(folder, filenames):
    if not folder.exists():
        return False
    missing = [name for name in filenames if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"{folder} 已存在，但缺少 {', '.join(missing)}")
    print(f"已存在，跳过 {folder.relative_to(ROOT)}")
    return True


def save_results(folder, splits, audit, cleaned=None):
    folder.mkdir(parents=True, exist_ok=False)
    if cleaned is not None:
        write_csv(folder / "cleaned.csv", cleaned)
    for name, rows in splits.items():
        write_csv(folder / f"{name}.csv", rows)
    audit["splits"] = {name: summarize(rows) for name, rows in splits.items()}
    (folder / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(folder.relative_to(ROOT), {name: len(rows) for name, rows in splits.items()})


def prepare_trofi_and_moh():
    for name, filename in FILES.items():
        folder = OUTPUT / name
        if output_exists(folder, ("cleaned.csv", "train.csv", "dev.csv", "test.csv")):
            continue
        cleaned, audit = clean_dataset(name, read_dataset(ROOT / "data" / filename))
        splits = split_rows(cleaned, SEED, "sentence")
        save_results(folder, splits, audit, cleaned)


def load_gold(path, expected_count, has_header=False):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if has_header and next(reader) != ["id", "label"]:
            raise ValueError(f"标签文件表头有误 {path}")
        gold = {}
        for token_id, label in reader:
            if token_id in gold or label not in {"0", "1"}:
                raise ValueError(f"重复 ID 或非法标签 {token_id}")
            gold[token_id] = label
    if len(gold) != expected_count:
        raise ValueError(f"{path.name}: 预期 {expected_count} 条，实际 {len(gold)} 条")
    return gold


def select_verbs(path, gold):
    selected = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, strict=True):
            token_id = row["id"]
            if token_id not in gold:
                continue
            index = int(token_id.rsplit("_", 1)[1]) - 1
            tokens = row["sentence"].split()
            if (
                token_id in selected
                or not 0 <= index < len(tokens)
                or tokens[index] != row["word"]
                or row["label"] != gold[token_id]
            ):
                raise ValueError(f"{path.name}: 样本校验失败 {token_id}")
            selected[token_id] = {
                "verb": row["word"],
                "sentence": row["sentence"],
                "verb_idx": index,
                "label": gold[token_id],
            }
    if selected.keys() != gold.keys():
        missing = gold.keys() - selected.keys()
        raise ValueError(f"{path.name}: 缺少 {len(missing)} 条官方样本")
    return list(selected.values())


def prepare_vua():
    folder = OUTPUT / "VUA"
    if output_exists(folder, ("train.csv", "dev.csv", "test.csv", "audit.json")):
        return

    train_gold = load_gold(TRAIN_GOLD, 17240)
    test_gold = load_gold(VUA / "verb_tokens_test_gold_labels.csv", 5873, has_header=True)
    if train_gold.keys() & test_gold.keys():
        raise ValueError("VUA 官方训练和测试 ID 有重合")

    source_train = select_verbs(VUA / "VUA_train.csv", train_gold)
    test = select_verbs(VUA / "VUA_verb_test.csv", test_gold)
    test_sentences = {sentence_key(row) for row in test}

    candidates = [row for row in source_train if sentence_key(row) not in test_sentences]
    targets = defaultdict(list)
    for row in candidates:
        targets[target_key(row)].append(row)

    sentence_groups = defaultdict(list)
    duplicates = conflicts = 0
    for rows in targets.values():
        if len({row["label"] for row in rows}) > 1:
            conflicts += len(rows)
            continue
        sentence_groups[sentence_key(rows[0])].append(rows[0])
        duplicates += len(rows) - 1

    groups = list(sentence_groups.values())
    random.Random(SEED).shuffle(groups)
    dev_target = round(sum(map(len, groups)) * 0.1)
    train, dev = [], []
    for group in groups:
        (dev if len(dev) < dev_target else train).extend(group)

    splits = {"train": train, "dev": dev, "test": test}
    seen = set()
    for name, rows in splits.items():
        sentences = {sentence_key(row) for row in rows}
        if not rows or seen & sentences:
            raise ValueError(f"VUA {name}: 集合为空或句子跨集合重合")
        seen.update(sentences)

    save_results(folder, splits, {
        "seed": SEED,
        "group_by": "sentence",
        "official_train_rows": len(source_train),
        "removed_train_test_sentence_overlap": len(source_train) - len(candidates),
        "removed_train_duplicates": duplicates,
        "removed_train_conflicting_labels": conflicts,
    })


if __name__ == "__main__":
    prepare_vua()
    prepare_trofi_and_moh()