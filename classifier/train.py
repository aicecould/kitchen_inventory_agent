"""Train FastText and select a recall-constrained decision threshold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from classifier.data import (
    DEFAULT_DATASET_DIR,
    datasets_sha256,
    discover_dataset_csvs,
    load_dataset,
    stratified_split,
    write_fasttext_file,
)
from classifier.evaluate import calculate_metrics, choose_threshold
from classifier.model import positive_scores
from classifier.preprocessing import PREPROCESSING_VERSION

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = DEFAULT_DATASET_DIR
DEFAULT_ARTIFACTS = ROOT_DIR / "classifier" / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the food query classifier")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV file or directory; directories load all valid CSVs",
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--fasttext-binary", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-recall", type=float, default=0.98)
    parser.add_argument("--minimum-threshold", type=float, default=0.1)
    return parser.parse_args()


def locate_fasttext_binary(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["FASTTEXT_BINARY"])
        if os.environ.get("FASTTEXT_BINARY")
        else None,
        Path(shutil.which("fasttext")) if shutil.which("fasttext") else None,
        ROOT_DIR / ".run" / "fasttext-linux-build" / "fasttext",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "FastText CLI not found. Pass --fasttext-binary or set FASTTEXT_BINARY."
    )


def train(
    *,
    dataset_path: Path,
    artifacts_dir: Path,
    fasttext_binary: Path,
    seed: int,
    minimum_recall: float,
    minimum_threshold: float = 0.1,
) -> dict[str, object]:
    records = load_dataset(dataset_path)
    split = stratified_split(records, seed=seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    work_dir = artifacts_dir / "training"
    work_dir.mkdir(parents=True, exist_ok=True)

    training_options = [
        "-lr",
        "0.2",
        "-epoch",
        "100",
        "-wordNgrams",
        "2",
        "-dim",
        "100",
        "-minCount",
        "1",
        "-minn",
        "1",
        "-maxn",
        "4",
        "-bucket",
        "50000",
        "-loss",
        "softmax",
        "-thread",
        "1",
        "-verbose",
        "2",
    ]

    deployment_train_path = work_dir / "deployment_train.txt"
    write_fasttext_file(
        [*split.train, *split.validation],
        deployment_train_path,
    )
    model_prefix = artifacts_dir / "food_query"
    subprocess.run(
        [
            str(fasttext_binary),
            "supervised",
            "-input",
            str(deployment_train_path),
            "-output",
            str(model_prefix),
            *training_options,
        ],
        check=True,
        cwd=ROOT_DIR,
    )

    model_path = model_prefix.with_suffix(".bin")
    import fasttext

    deployment_model = fasttext.load_model(str(model_path))
    validation_labels = [record.label for record in split.validation]
    validation_scores = positive_scores(
        deployment_model,
        [record.text for record in split.validation],
    )
    threshold, validation_metrics = choose_threshold(
        validation_labels,
        validation_scores,
        minimum_recall=minimum_recall,
        minimum_threshold=minimum_threshold,
    )
    test_labels = [record.label for record in split.test]
    test_scores = positive_scores(
        deployment_model,
        [record.text for record in split.test],
    )
    test_metrics = calculate_metrics(test_labels, test_scores, threshold)
    resolved_dataset = dataset_path.resolve()
    try:
        dataset_display_path = str(resolved_dataset.relative_to(ROOT_DIR))
    except ValueError:
        dataset_display_path = str(resolved_dataset)

    dataset_paths = discover_dataset_csvs(dataset_path)
    dataset_files: list[str] = []
    for path in dataset_paths:
        try:
            dataset_files.append(str(path.relative_to(ROOT_DIR)))
        except ValueError:
            dataset_files.append(str(path))

    metadata: dict[str, object] = {
        "model_type": "fasttext-supervised",
        "domain": "recipe_and_ingredient_queries",
        "positive_label": "__label__1",
        "threshold": threshold,
        "minimum_validation_recall": minimum_recall,
        "minimum_threshold": minimum_threshold,
        "preprocessing": "nfkc_lowercase_jieba",
        "preprocessing_version": PREPROCESSING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_display_path,
        "dataset_sha256": datasets_sha256(dataset_paths),
        "dataset_files": dataset_files,
        "seed": seed,
        "split_counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "deployment_train": len(split.train) + len(split.validation),
        },
        "validation_metrics": asdict(validation_metrics),
        "test_metrics": asdict(test_metrics),
        "training_parameters": {
            "lr": 0.2,
            "epoch": 100,
            "wordNgrams": 2,
            "dim": 100,
            "minCount": 1,
            "minn": 1,
            "maxn": 4,
            "bucket": 50000,
            "loss": "softmax",
            "thread": 1,
        },
    }
    metadata_path = artifacts_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluation_path = artifacts_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "validation": asdict(validation_metrics),
                "test": asdict(test_metrics),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = train(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts,
        fasttext_binary=locate_fasttext_binary(args.fasttext_binary),
        seed=args.seed,
        minimum_recall=args.minimum_recall,
        minimum_threshold=args.minimum_threshold,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
