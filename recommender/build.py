"""One-shot offline builder for the local recipe database."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from recommender.ingredients import parse_ingredient_lines


class BuildInputError(ValueError):
    """The CSV source cannot be used to build recipe artifacts."""


REQUIRED_COLUMNS = (
    "Title",
    "Ingredients",
    "Instructions",
    "Image_Name",
    "Cleaned_Ingredients",
)


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
