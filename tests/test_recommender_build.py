import csv
import sqlite3
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import pytest

from recommender.build import (
    BuildInputError,
    build_artifacts,
    build_faiss,
    build_sqlite,
    publish_artifacts,
    validate_sqlite,
)
from recommender.model_config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION


def write_recipe_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "Title",
                "Ingredients",
                "Instructions",
                "Image_Name",
                "Cleaned_Ingredients",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "": "0",
                "Title": "Tomato Egg",
                "Ingredients": "['2 tomatoes', '2 eggs']",
                "Instructions": "Cook tomato and egg.",
                "Image_Name": "tomato-egg",
                "Cleaned_Ingredients": "['2 tomatoes', '2 eggs']",
            }
        )
        writer.writerow(
            {
                "": "1",
                "Title": "Peanut Noodles",
                "Ingredients": "['noodles', 'peanuts']",
                "Instructions": "Mix noodles and peanuts.",
                "Image_Name": "peanut-noodles",
                "Cleaned_Ingredients": "['noodles', 'peanuts']",
            }
        )


def test_build_sqlite_creates_versioned_fts_database(tmp_path: Path) -> None:
    source = tmp_path / "recipes.csv"
    destination = tmp_path / "recipes.db"
    write_recipe_csv(source)

    count = build_sqlite(source, destination)
    validate_sqlite(destination, expected_count=2)

    assert count == 2
    connection = sqlite3.connect(f"file:{destination}?mode=ro", uri=True)
    assert connection.execute("SELECT version FROM schema_versions").fetchone() == (1,)
    assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM recipes_fts").fetchone()[0] == 2
    assert (
        connection.execute(
            "SELECT rowid FROM recipes_fts WHERE recipes_fts MATCH ?",
            ('"tomato"',),
        ).fetchone()[0]
        == 1
    )
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM recipes")
    connection.close()


def test_build_sqlite_rejects_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text("Title,Instructions\nTomato Soup,Cook it.\n", encoding="utf-8")

    with pytest.raises(BuildInputError):
        build_sqlite(source, tmp_path / "invalid.db")


def test_build_sqlite_falls_back_when_cleaned_ingredients_are_empty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fallback.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "Title",
                "Ingredients",
                "Instructions",
                "Image_Name",
                "Cleaned_Ingredients",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "": "0",
                "Title": "Tomato",
                "Ingredients": "['tomatoes']",
                "Instructions": "Cook.",
                "Image_Name": "",
                "Cleaned_Ingredients": "[]",
            }
        )

    count = build_sqlite(source, tmp_path / "fallback.db")

    assert count == 1
    connection = sqlite3.connect(
        f"file:{tmp_path / 'fallback.db'}?mode=ro",
        uri=True,
    )
    row = connection.execute(
        "SELECT normalized_ingredients_json FROM recipes"
    ).fetchone()
    assert '"tomato"' in row[0]
    connection.close()


def test_main_pipeline_source_never_imports_build_logic() -> None:
    pipeline_source = Path("app/pipeline.py").read_text(encoding="utf-8")
    assert "recommender.build" not in pipeline_source
    assert "build_sqlite" not in pipeline_source


class FakeEncoder:
    model_id = EMBEDDING_MODEL
    model_revision = EMBEDDING_MODEL_REVISION

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % 3] = 1.0
        return vectors


class NaNFakeEncoder(FakeEncoder):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.full((len(texts), 3), np.nan, dtype=np.float32)


class WrongModelIdEncoder(FakeEncoder):
    model_id = "BAAI/wrong-model"


class WrongRevisionEncoder(FakeEncoder):
    model_revision = "wrong-revision"


def test_build_faiss_writes_id_mapped_index(tmp_path: Path) -> None:
    source = tmp_path / "recipes.csv"
    database = tmp_path / "recipes.db"
    index_path = tmp_path / "recipes.faiss"
    write_recipe_csv(source)
    build_sqlite(source, database)

    dimension, count = build_faiss(database, index_path, FakeEncoder())

    assert (dimension, count) == (3, 2)
    index = faiss.read_index(str(index_path))
    assert isinstance(index, faiss.IndexIDMap2)
    assert index.ntotal == 2
    assert index.d == 3
    query = np.zeros((1, 3), dtype=np.float32)
    query[0, 0] = 1.0
    distances, ids = index.search(query, 2)
    assert sorted(ids[0].tolist()) == [1, 2]


def test_build_faiss_rejects_nan_vectors(tmp_path: Path) -> None:
    source = tmp_path / "recipes.csv"
    database = tmp_path / "recipes.db"
    write_recipe_csv(source)
    build_sqlite(source, database)

    with pytest.raises(ValueError):
        build_faiss(database, tmp_path / "bad.faiss", NaNFakeEncoder())


def test_publish_artifacts_is_atomic_and_cleans_temp(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "recipes.db").write_text("old-db", encoding="utf-8")
    (output_dir / "recipes.faiss").write_text("old-faiss", encoding="utf-8")
    (output_dir / "metadata.json").write_text("old-metadata", encoding="utf-8")
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "recipes.db").write_text("new-db", encoding="utf-8")
    (temp_dir / "recipes.faiss").write_text("new-faiss", encoding="utf-8")

    with pytest.raises(RuntimeError):
        publish_artifacts(temp_dir, output_dir)

    assert (output_dir / "recipes.db").read_text(encoding="utf-8") == "old-db"
    assert (output_dir / "recipes.faiss").read_text(encoding="utf-8") == "old-faiss"
    assert (output_dir / "metadata.json").read_text(encoding="utf-8") == "old-metadata"
    assert not temp_dir.exists()


@pytest.mark.parametrize(
    "encoder_type",
    [WrongModelIdEncoder, WrongRevisionEncoder],
)
def test_build_artifacts_rejects_unverified_encoder_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder_type: type[FakeEncoder],
) -> None:
    source = tmp_path / "recipes.csv"
    write_recipe_csv(source)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "recipes.db").write_text("old-db", encoding="utf-8")
    (output_dir / "recipes.faiss").write_text("old-faiss", encoding="utf-8")
    (output_dir / "metadata.json").write_text("old-metadata", encoding="utf-8")
    monkeypatch.setattr(
        "recommender.build.LocalSentenceEncoder",
        lambda model_path, *, local_files_only: encoder_type(),
    )

    with pytest.raises(BuildInputError):
        build_artifacts(
            source=source,
            model_path=tmp_path / "model",
            output_dir=output_dir,
        )

    assert (output_dir / "recipes.db").read_text(encoding="utf-8") == "old-db"
    assert (output_dir / "recipes.faiss").read_text(encoding="utf-8") == "old-faiss"
    assert (output_dir / "metadata.json").read_text(encoding="utf-8") == "old-metadata"
