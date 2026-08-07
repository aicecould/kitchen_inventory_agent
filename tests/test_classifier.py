from pathlib import Path

import pytest

from classifier import train as classifier_train
from classifier.data import load_dataset, stratified_split
from classifier.evaluate import calculate_metrics, choose_threshold
from classifier.model import FoodQueryClassifier, positive_score
from classifier.preprocessing import normalize_text, preprocess_text


class FakeFastTextModel:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict(
        self, text: str | list[str], *, k: int = 1
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        assert isinstance(text, str)
        assert k == 2
        return (
            ("__label__1", "__label__0"),
            (self.positive_probability, 1 - self.positive_probability),
        )


def test_preprocessing_normalizes_and_segments_chinese() -> None:
    assert normalize_text("  APPLE　鸡蛋  ") == "apple 鸡蛋"
    assert preprocess_text("番茄炒鸡蛋怎么做") == "番茄 炒鸡蛋 怎么 做"


def test_dataset_accepts_text_column_and_stratifies(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text(
        "text,label\n番茄怎么炒,1\n土豆怎么煮,1\n鸡蛋怎么做,1\n"
        "Python怎么装,0\n天气怎么样,0\n手机怎么修,0\n",
        encoding="utf-8",
    )

    records = load_dataset(path)
    split = stratified_split(records)

    assert len(records) == 6
    assert {record.label for record in split.train} == {0, 1}
    assert {record.label for record in split.validation} == {0, 1}
    assert {record.label for record in split.test} == {0, 1}


def test_dataset_rejects_conflicting_duplicate_labels(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text(
        "query,label\n番茄怎么炒,1\n番茄怎么炒,0\n天气,0\n鸡蛋,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(path)


def test_threshold_selection_prioritizes_requested_recall() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.7, 0.2]

    threshold, metrics = choose_threshold(labels, scores, minimum_recall=1.0)

    assert threshold == 0.4
    assert metrics.recall == 1.0
    assert metrics.false_positive_rate == 0.5
    assert calculate_metrics(labels, scores, threshold) == metrics


def test_runtime_classifier_uses_positive_probability_threshold() -> None:
    model = FakeFastTextModel(0.35)
    classifier = FoodQueryClassifier(model, threshold=0.3)

    result = classifier.classify("西红柿炒鸡蛋怎么做")

    assert positive_score(model, "西红柿炒鸡蛋怎么做") == pytest.approx(0.35)
    assert result.is_related is True
    assert result.positive_score == pytest.approx(0.35)


def test_training_finds_linux_build_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / ".run" / "fasttext-linux-build" / "fasttext"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(classifier_train, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(classifier_train.shutil, "which", lambda _: None)
    monkeypatch.delenv("FASTTEXT_BINARY", raising=False)

    located = classifier_train.locate_fasttext_binary(None)

    assert located == binary.resolve()
