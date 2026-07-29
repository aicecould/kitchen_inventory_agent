"""Dataset loading, validation, deduplication and deterministic splitting."""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from classifier.preprocessing import normalize_text, preprocess_text


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    text: str
    label: int


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: list[LabeledQuery]
    validation: list[LabeledQuery]
    test: list[LabeledQuery]


def load_dataset(path: Path) -> list[LabeledQuery]:
    """Load a two-column CSV accepting either `query` or `text`."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        text_column = "query" if "query" in headers else "text" if "text" in headers else None
        if text_column is None or "label" not in headers:
            raise ValueError("Dataset must contain query/text and label columns")

        unique: dict[str, LabeledQuery] = {}
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            raw_label = (row.get("label") or "").strip()
            if not text:
                raise ValueError(f"Dataset row {row_number} has an empty query")
            if raw_label not in {"0", "1"}:
                raise ValueError(
                    f"Dataset row {row_number} has invalid label {raw_label!r}"
                )

            key = normalize_text(text)
            label = int(raw_label)
            existing = unique.get(key)
            if existing is not None and existing.label != label:
                raise ValueError(
                    f"Dataset contains conflicting labels for query {text!r}"
                )
            unique.setdefault(key, LabeledQuery(text=text, label=label))

    records = list(unique.values())
    labels = {record.label for record in records}
    if labels != {0, 1}:
        raise ValueError("Dataset must contain both label 0 and label 1")
    return records


def stratified_split(
    records: list[LabeledQuery],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> DatasetSplit:
    """Split each class independently so all partitions remain balanced."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1")

    partitions: dict[str, list[LabeledQuery]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in (0, 1):
        class_records = [record for record in records if record.label == label]
        if len(class_records) < 3:
            raise ValueError("Each class needs at least three examples")
        random.Random(seed + label).shuffle(class_records)

        train_size = min(
            max(1, int(len(class_records) * train_ratio)),
            len(class_records) - 2,
        )
        validation_size = min(
            max(1, int(len(class_records) * validation_ratio)),
            len(class_records) - train_size - 1,
        )
        train_end = train_size
        validation_end = train_end + validation_size

        partitions["train"].extend(class_records[:train_end])
        partitions["validation"].extend(class_records[train_end:validation_end])
        partitions["test"].extend(class_records[validation_end:])

    for offset, values in enumerate(partitions.values()):
        random.Random(seed + 100 + offset).shuffle(values)
    return DatasetSplit(**partitions)


def write_fasttext_file(records: list[LabeledQuery], path: Path) -> None:
    """Write UTF-8 supervised FastText input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            processed = preprocess_text(record.text)
            if not processed:
                raise ValueError(f"Query became empty after preprocessing: {record.text!r}")
            handle.write(f"__label__{record.label} {processed}\n")


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
