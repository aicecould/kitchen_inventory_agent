"""One-shot offline builder for the local recipe database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from recommender.embeddings import DocumentEncoder, LocalSentenceEncoder
from recommender.ingredients import parse_ingredient_lines
from recommender.models import BuildMetadata
from recommender.model_config import (
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
)


class BuildInputError(ValueError):
    """The CSV source cannot be used to build recipe artifacts."""


REQUIRED_COLUMNS = (
    "Title",
    "Ingredients",
    "Instructions",
    "Image_Name",
    "Cleaned_Ingredients",
)
SOURCE_COMMIT = "c5a1e8dab69d46841d914be1f20cca6fbe4b6260"


def build_sqlite(source: Path, destination: Path) -> int:
    """Import a recipe CSV into a versioned SQLite/FTS5 database."""
    if destination.exists():
        destination.unlink()
    connection = sqlite3.connect(destination)
    try:
        _create_schema(connection)
        count = 0
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
            if missing:
                raise BuildInputError(f"Missing required columns: {', '.join(missing)}")
            for row in reader:
                title = (row.get("Title") or "").strip()
                instructions = (row.get("Instructions") or "").strip()
                ingredient_lines = parse_ingredient_lines(
                    row.get("Cleaned_Ingredients") or ""
                )
                if not ingredient_lines:
                    ingredient_lines = parse_ingredient_lines(
                        row.get("Ingredients") or ""
                    )
                if not title or not instructions or not ingredient_lines:
                    continue
                connection.execute(
                    """
                    INSERT INTO recipes (
                        title,
                        ingredient_lines_json,
                        normalized_ingredients_json,
                        ingredient_search_text,
                        instructions,
                        image_name,
                        source_dataset
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        title,
                        json.dumps(ingredient_lines, ensure_ascii=False),
                        json.dumps(ingredient_lines, ensure_ascii=False),
                        " ".join(ingredient_lines),
                        instructions,
                        (row.get("Image_Name") or "").strip() or None,
                        "josephrmartinez/recipe-dataset",
                    ),
                )
                count += 1
        connection.execute("INSERT INTO recipes_fts(recipes_fts) VALUES ('rebuild')")
        connection.commit()
        if count <= 0:
            raise BuildInputError("No valid recipe rows found")
        return count
    finally:
        connection.close()


def validate_sqlite(path: Path, expected_count: int) -> None:
    """Validate built SQLite schema, FTS contents, and row JSON."""
    connection = sqlite3.connect(path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        version = connection.execute("SELECT version FROM schema_versions").fetchone()
        if version != (1,):
            raise ValueError("Unexpected schema version")
        recipe_count = connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM recipes_fts").fetchone()[0]
        if recipe_count != expected_count or fts_count != expected_count:
            raise ValueError("Recipe and FTS row counts differ from expected count")
        for (raw_json,) in connection.execute(
            "SELECT normalized_ingredients_json FROM recipes"
        ):
            values = json.loads(raw_json)
            if not isinstance(values, list) or not values:
                raise ValueError("Recipe has invalid normalized ingredients JSON")
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError("Recipe has a non-string normalized ingredient")
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE schema_versions (
            version INTEGER PRIMARY KEY
        );

        CREATE TABLE recipes (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            ingredient_lines_json TEXT NOT NULL,
            normalized_ingredients_json TEXT NOT NULL,
            ingredient_search_text TEXT NOT NULL,
            instructions TEXT NOT NULL,
            image_name TEXT,
            source_dataset TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE recipes_fts USING fts5(
            title,
            ingredient_search_text,
            instructions,
            content='recipes',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );

        INSERT INTO schema_versions (version) VALUES (1);
        """
    )


def build_faiss(
    db_path: Path,
    destination: Path,
    encoder: DocumentEncoder,
) -> tuple[int, int]:
    """Build a FAISS IndexIDMap2 keyed by SQLite recipe IDs."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT id, title, ingredient_search_text, instructions
            FROM recipes
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("Cannot build FAISS from an empty recipe database")
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    texts = [
        f"Title: {row[1]}\nIngredients: {row[2]}\nInstructions: {row[3]}"
        for row in rows
    ]
    vectors = encoder.encode_documents(texts)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError("Encoder returned vectors incompatible with recipe rows")
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned NaN or Inf vectors")
    dimension = vectors.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    index.add_with_ids(vectors.astype(np.float32), ids)
    faiss.write_index(index, str(destination))
    return dimension, len(rows)


def publish_artifacts(temp_dir: Path, output_dir: Path) -> None:
    """Atomically publish validated artifacts, replacing metadata last."""
    try:
        expected = ("recipes.db", "recipes.faiss", "metadata.json")
        for name in expected:
            if not (temp_dir / name).is_file():
                raise RuntimeError(f"Temporary artifact is missing: {name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir / "recipes.db", output_dir / "recipes.db")
        os.replace(temp_dir / "recipes.faiss", output_dir / "recipes.faiss")
        os.replace(temp_dir / "metadata.json", output_dir / "metadata.json")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_artifacts(source: Path, model_path: Path, output_dir: Path) -> None:
    """Build all runtime artifacts and publish them atomically."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as raw_temp:
        temp_dir = Path(raw_temp)
        database = temp_dir / "recipes.db"
        faiss_path = temp_dir / "recipes.faiss"
        metadata_path = temp_dir / "metadata.json"
        recipe_count = build_sqlite(source, database)
        validate_sqlite(database, expected_count=recipe_count)
        encoder = LocalSentenceEncoder(model_path, local_files_only=True)
        dimension, encoded_count = build_faiss(database, faiss_path, encoder)
        if encoded_count != recipe_count:
            raise ValueError("FAISS and SQLite recipe counts differ")
        metadata = BuildMetadata(
            source_commit=SOURCE_COMMIT,
            source_sha256=_sha256_file(source),
            sqlite_sha256=_sha256_file(database),
            license="CC BY-SA 3.0",
            embedding_model=EMBEDDING_MODEL,
            embedding_model_revision=EMBEDDING_MODEL_REVISION,
            embedding_max_seq_length=EMBEDDING_MAX_SEQ_LENGTH,
            embedding_dimension=dimension,
            recipe_count=recipe_count,
            built_at=datetime.now(UTC),
        )
        metadata_path.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        publish_artifacts(temp_dir, output_dir)
    print(
        f"Published {recipe_count} recipes, dimension {dimension} to {output_dir}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local recipe SQLite/FTS5, FAISS, and metadata artifacts."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_artifacts(args.source, args.model, args.output_dir)


if __name__ == "__main__":
    main()
