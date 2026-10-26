"""Convert downloaded MASSIVE zh-CN JSONL into the project's binary CSVs.

Cooking intents are positive; other intents (including takeaway) are negative.
The 500-row file is a subset of the 1,000-row file, so merging adds no extra
negative examples. The default sampling seed is 42.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from classifier.preprocessing import normalize_text


POSITIVE_INTENTS = frozenset({"cooking_query", "cooking_recipe"})


def prepare(source: Path, output_dir: Path, *, seed: int = 42) -> dict:
    records: dict[str, tuple[str, int]] = {}
    source_bytes = source.read_bytes()
    raw_count = 0
    for line in source_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("locale") != "zh-CN":
            raise ValueError("Expected only MASSIVE zh-CN records")
        text = row["utt"].strip()
        intent = row["intent"]
        if not text or not isinstance(intent, str) or not intent:
            raise ValueError("Empty utterance or intent")
        label = int(intent in POSITIVE_INTENTS)
        key = normalize_text(text)
        if key in records and records[key][1] != label:
            raise ValueError(f"Conflicting labels for {text!r}")
        records.setdefault(key, (text, label))
        raw_count += 1

    positive = [r for r in records.values() if r[1] == 1]
    negative = [r for r in records.values() if r[1] == 0]
    if not positive or len(negative) < 1000:
        raise ValueError("Need positive records and at least 1000 unique negatives")
    sampled = random.Random(seed).sample(negative, 1000)
    outputs = {
        "massive_cook.csv": positive,
        "massive_non_cook_1000.csv": sampled,
        "massive_non_cook_500.csv": sampled[:500],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": seed,
        "raw_count": raw_count,
        "unique_labels": dict(Counter(r[1] for r in records.values())),
        "positive_intents": sorted(POSITIVE_INTENTS),
        "sampling": "1000 unique negatives; 500-row file is their first 500",
        "outputs": {},
    }
    for name, values in outputs.items():
        path = output_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "label"])
            writer.writerows(values)
        report["outputs"][name] = {
            "rows": len(values),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output_dir / "massive-preparation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("classifier/dataset/massive/zh-CN.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("classifier/dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, seed=args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
