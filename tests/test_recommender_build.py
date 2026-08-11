import csv
import sqlite3
from pathlib import Path

import pytest

from recommender.build import BuildInputError, build_sqlite, validate_sqlite


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
