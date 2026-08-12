from pathlib import Path

import pytest

from classifier.data import (
    discover_dataset_csvs,
    load_dataset,
    load_dataset_with_optional_extra,
    load_merged_datasets,
)


def _write_csv(path: Path, rows: str) -> Path:
    path.write_text(f"text,label\n{rows}", encoding="utf-8")
    return path


def _write_gbk_tsv(path: Path, rows: str) -> Path:
    path.write_bytes(f"text\tlabel\r\n{rows}".encode("gbk"))
    return path


def test_load_merged_datasets_deduplicates_and_preserves_records(
    tmp_path: Path,
) -> None:
    primary = _write_csv(
        tmp_path / "primary.csv",
        "番茄炒蛋怎么做,1\n鸡蛋怎么煎,1\nPython怎么装,0\n天气怎么样,0\n",
    )
    extra = _write_csv(
        tmp_path / "extra.csv",
        "番茄炒蛋怎么做,1\n牛肉怎么做,1\n股票怎么看,0\n",
    )

    records = load_merged_datasets([primary, extra])

    assert len(records) == 6
    assert {record.label for record in records} == {0, 1}
    assert sum(record.label == 1 for record in records) == 3
    assert sum(record.label == 0 for record in records) == 3


def test_load_merged_datasets_rejects_cross_file_label_conflict(
    tmp_path: Path,
) -> None:
    primary = _write_csv(
        tmp_path / "primary.csv",
        "番茄炒蛋怎么做,1\n鸡蛋怎么煎,1\nPython怎么装,0\n天气怎么样,0\n",
    )
    extra = _write_csv(
        tmp_path / "extra.csv",
        "番茄炒蛋怎么做,0\n牛肉怎么做,1\n股票怎么看,0\n",
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        load_merged_datasets([primary, extra])


def test_load_dataset_with_optional_extra_skips_missing_file(
    tmp_path: Path,
) -> None:
    primary = _write_csv(
        tmp_path / "primary.csv",
        "番茄炒蛋怎么做,1\n鸡蛋怎么煎,1\nPython怎么装,0\n天气怎么样,0\n",
    )

    records = load_dataset_with_optional_extra(
        primary,
        tmp_path / "massive_cook_sample.csv",
    )

    assert records == load_dataset(primary)


def test_load_dataset_with_optional_extra_merges_existing_file(
    tmp_path: Path,
) -> None:
    primary = _write_csv(
        tmp_path / "primary.csv",
        "番茄炒蛋怎么做,1\n鸡蛋怎么煎,1\nPython怎么装,0\n天气怎么样,0\n",
    )
    extra = _write_csv(
        tmp_path / "extra.csv",
        "番茄炒蛋怎么做,1\n牛肉怎么做,1\n股票怎么看,0\n",
    )

    records = load_dataset_with_optional_extra(primary, extra)

    assert len(records) == 6


def test_load_dataset_reads_all_valid_csvs_in_directory(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_csv(
        dataset_dir / "semantic.csv",
        "番茄炒蛋怎么做,1\nPython怎么装,0\n",
    )
    _write_gbk_tsv(
        dataset_dir / "massive_cook.csv",
        "牛肉怎么做\t1\n",
    )
    _write_gbk_tsv(
        dataset_dir / "massive_negative.csv",
        "Python怎么装\t0\n天气怎么样\t0\n",
    )
    (dataset_dir / "ignored.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")

    records = load_dataset(dataset_dir)

    assert len(records) == 4
    assert {record.label for record in records} == {0, 1}
    assert sum(record.label == 1 for record in records) == 2
    assert sum(record.label == 0 for record in records) == 2


def test_load_dataset_directory_rejects_cross_file_conflict(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_csv(
        dataset_dir / "primary.csv",
        "番茄炒蛋怎么做,1\nPython怎么装,0\n",
    )
    _write_gbk_tsv(
        dataset_dir / "conflict.csv",
        "番茄炒蛋怎么做\t0\n天气怎么样\t0\n",
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(dataset_dir)


def test_discover_dataset_csvs_skips_non_conforming_files(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    valid = _write_csv(
        dataset_dir / "valid.csv",
        "番茄炒蛋怎么做,1\nPython怎么装,0\n",
    )
    _write_gbk_tsv(dataset_dir / "negative.csv", "天气怎么样\t0\n")
    (dataset_dir / "not_a_dataset.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")

    paths = discover_dataset_csvs(dataset_dir)

    assert paths == sorted([valid, dataset_dir / "negative.csv"])


def test_load_merged_datasets_accepts_single_label_gbk_tab_files(
    tmp_path: Path,
) -> None:
    positive = _write_gbk_tsv(tmp_path / "positive.csv", "牛肉怎么做\t1\n")
    negative = _write_gbk_tsv(tmp_path / "negative.csv", "天气怎么样\t0\n")

    records = load_merged_datasets([positive, negative])

    assert len(records) == 2
    assert {record.label for record in records} == {0, 1}
