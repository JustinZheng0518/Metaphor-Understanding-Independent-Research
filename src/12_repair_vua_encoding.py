
"""
Repair known UTF-8 mojibake in VUA CSV files.

- Back up originals
- Repair malformed punctuation
- Preserve row order, labels and target indices
- Validate before overwriting

Does not modify model checkpoints or results.
"""

from pathlib import Path
from datetime import datetime
from io import StringIO
import re
import shutil

import pandas as pd


DATA_DIR = Path("data/VUA")

FILES = {
    "VUA_formatted_train.csv": 15516,
    "VUA_formatted_val.csv": 1724,
    "VUA_formatted_test.csv": 5873,
}

# Matches strings such as:
# â + U+0080 + U+0098  -> ‘
# â + U+0080 + U+0094  -> —
# â + U+0080 + U+00A6  -> …

PATTERN = re.compile(r"â[\x80-\xBF][\x80-\xBF]")


def repair_match(match):
    corrupted = match.group(0)
    return corrupted.encode("latin1").decode("utf-8")


def validate(original, repaired, expected_rows, filename):
    old_df = pd.read_csv(StringIO(original))
    new_df = pd.read_csv(StringIO(repaired))

    if len(old_df) != expected_rows:
        raise ValueError(
            f"{filename}: unexpected original row count"
        )

    if len(new_df) != expected_rows:
        raise ValueError(
            f"{filename}: row count changed"
        )

    if list(old_df.columns) != list(new_df.columns):
        raise ValueError(
            f"{filename}: columns changed"
        )

    # The repair must not change dataset identities,
    # target positions, or gold labels.
    key_columns = [
        "text_idx",
        "sentence_idx",
        "verb_idx",
        "label",
    ]

    pd.testing.assert_frame_equal(
        old_df[key_columns],
        new_df[key_columns],
        check_dtype=False,
    )

    return len(new_df)


def main():
    print("===== VUA ENCODING REPAIR =====")

    prepared = {}

    # Validate all files before changing any of them.
    for filename, expected_rows in FILES.items():

        path = DATA_DIR / filename

        original_bytes = path.read_bytes()
        original = original_bytes.decode("utf-8")

        repaired, count = PATTERN.subn(
            repair_match,
            original,
        )

        rows = validate(
            original,
            repaired,
            expected_rows,
            filename,
        )

        prepared[filename] = {
            "original_bytes": original_bytes,
            "repaired_bytes": repaired.encode("utf-8"),
            "count": count,
        }

        print(
            f"{filename}: "
            f"{rows} rows, "
            f"{count} replacements"
        )

    if all(x["count"] == 0 for x in prepared.values()):
        print("No matching corruption found.")
        print("No files modified.")
        return

    # Never overwrite the original backup.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup_dir = (
        DATA_DIR / f"encoding_backup_{timestamp}"
    )

    backup_dir.mkdir(parents=True)

    print("\n===== BACKUP =====")

    for filename, item in prepared.items():
        backup_path = backup_dir / filename

        backup_path.write_bytes(
            item["original_bytes"]
        )

        print("Backed up:", backup_path)

    print("\n===== APPLY REPAIR =====")

    for filename, item in prepared.items():
        path = DATA_DIR / filename

        if item["count"] > 0:
            # Temporary file, then atomic replacement.
            temp_path = path.with_name(
                path.name + ".repair_tmp"
            )

            temp_path.write_bytes(
                item["repaired_bytes"]
            )

            temp_path.replace(path)

        print("Checked:", path)

    print("\n===== FINAL VERIFICATION =====")

    for filename, expected_rows in FILES.items():
        path = DATA_DIR / filename

        content = path.read_text(
            encoding="utf-8"
        )

        df = pd.read_csv(
            path,
            encoding="utf-8"
        )

        assert len(df) == expected_rows

        remaining = len(PATTERN.findall(content))

        assert remaining == 0, (
            f"Remaining corruption in {filename}"
        )

        print(
            f"{filename}: "
            f"{len(df)} rows, "
            f"{remaining} remaining patterns"
        )

    print("\nREPAIR COMPLETE")
    print("Originals saved in:", backup_dir)
    print("Use encoding='utf-8' from now on.")


if __name__ == "__main__":
    main()
