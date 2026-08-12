"""Dataset loading, validation, deduplication and deterministic splitting."""

from __future__ import annotations

import csv
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path

from classifier.preprocessing import normalize_text, preprocess_text

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
CSV_ENCODINGS = ("utf-8-sig", "gbk", "gb18030")
CSV_DELIMITERS = ("\t", ",", ";")


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
    """Load one CSV, or merge all valid CSVs under a directory."""
    if path.is_dir():
        return load_merged_datasets([path])

    records = _load_csv_file(path)
    _require_both_labels(records)
    return records


def discover_dataset_csvs(path: Path) -> list[Path]:
    """Return valid `text/query` + `label` CSV files for a file or directory."""
    resolved = path.resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    paths = sorted(resolved.rglob("*.csv"))
    valid = [candidate for candidate in paths if _dataset_headers(candidate)]
    if not valid:
        raise ValueError(f"No valid dataset CSVs found in {path}")
    return valid


def load_merged_datasets(paths: list[Path]) -> list[LabeledQuery]:
    """Load multiple CSVs and reject conflicting labels across files."""
    if not paths:
        raise ValueError("At least one dataset path is required")

    resolved_paths: list[Path] = []
    for path in paths:
        resolved_paths.extend(discover_dataset_csvs(path))
    if not resolved_paths:
        raise ValueError("At least one dataset path is required")

    unique: dict[str, LabeledQuery] = {}
    for path in resolved_paths:
        for record in _load_csv_file(path):
            key = normalize_text(record.text)
            existing = unique.get(key)
            if existing is not None and existing.label != record.label:
                raise ValueError(
                    f"Datasets contain conflicting labels for query {record.text!r}"
                )
            unique.setdefault(key, record)

    records = list(unique.values())
    _require_both_labels(records)
    return records


def load_dataset_with_optional_extra(
    primary: Path,
    extra: Path | None = None,
) -> list[LabeledQuery]:
    """Load primary and optional extra file/directory when the extra path exists."""
    paths = [primary]
    if extra is not None and extra.exists():
        paths.append(extra)
    return load_merged_datasets(paths)


def datasets_sha256(paths: list[Path]) -> str:
    """Hash resolved dataset file paths and their contents."""
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(dataset_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_headers(path: Path) -> set[str] | None:
    try:
        _, headers, _ = _read_csv_table(path)
    except (ValueError, csv.Error):
        return None
    if ("query" in headers or "text" in headers) and "label" in headers:
        return headers
    return None


def _load_csv_file(path: Path) -> list[LabeledQuery]:
    text, headers, delimiter = _read_csv_table(path)
    if not _dataset_headers(path):
        raise ValueError(f"Dataset must contain query/text and label columns: {path}")

    text_column = "query" if "query" in headers else "text"
    unique: dict[str, LabeledQuery] = {}
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row_number, row in enumerate(reader, start=2):
        text_value = (row.get(text_column) or "").strip()
        raw_label = (row.get("label") or "").strip()
        if not text_value:
            raise ValueError(f"Dataset row {row_number} has an empty query")
        if raw_label not in {"0", "1"}:
            raise ValueError(
                f"Dataset row {row_number} has invalid label {raw_label!r}"
            )

        key = normalize_text(text_value)
        label = int(raw_label)
        existing = unique.get(key)
        if existing is not None and existing.label != label:
            raise ValueError(
                f"Dataset contains conflicting labels for query {text_value!r}"
            )
        unique.setdefault(key, LabeledQuery(text=text_value, label=label))
    return list(unique.values())


def _read_csv_table(path: Path) -> tuple[str, set[str], str]:
    text = _decode_csv_text(path)
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = _detect_delimiter(first_line)
    headers = set(
        csv.DictReader(io.StringIO(text), delimiter=delimiter).fieldnames or ()
    )
    return text, headers, delimiter


def _decode_csv_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Dataset {path} has unsupported encoding")


def _detect_delimiter(header: str) -> str:
    for delimiter in CSV_DELIMITERS:
        if delimiter in header:
            return delimiter
    return ","


def _require_both_labels(records: list[LabeledQuery]) -> None:
    labels = {record.label for record in records}
    if labels != {0, 1}:
        raise ValueError("Dataset must contain both label 0 and label 1")


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
