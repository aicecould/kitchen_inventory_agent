"""Read-only access to the locally built SQLite recipe index."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from recommender.models import RecommenderArtifactsError, StoredRecipe


class RecipeRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open_readonly(cls, path: Path) -> "RecipeRepository":
        if not path.is_file():
            raise RecommenderArtifactsError(f"Recipe database is missing: {path}")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return cls(connection)

    def fts_search(self, match_query: str, limit: int = 100) -> list[int]:
        rows = self._connection.execute(
            """
            SELECT rowid
            FROM recipes_fts
            WHERE recipes_fts MATCH ?
            ORDER BY bm25(recipes_fts), rowid
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
        return [row[0] for row in rows]

    def count_recipes(self) -> int:
        return self._connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]

    def get_many(self, recipe_ids: Sequence[int]) -> dict[int, StoredRecipe]:
        if not recipe_ids:
            return {}
        placeholders = ", ".join("?" for _ in recipe_ids)
        rows = self._connection.execute(
            f"""
            SELECT
                id,
                title,
                ingredient_lines_json,
                normalized_ingredients_json,
                ingredient_search_text,
                instructions,
                image_name
            FROM recipes
            WHERE id IN ({placeholders})
            """,
            list(recipe_ids),
        ).fetchall()
        recipes: dict[int, StoredRecipe] = {}
        for row in rows:
            recipe_id, title, lines_json, normalized_json, search_text, instructions, image_name = row
            recipes[recipe_id] = StoredRecipe(
                id=recipe_id,
                title=title,
                ingredient_lines=tuple(json.loads(lines_json)),
                normalized_ingredients=frozenset(json.loads(normalized_json)),
                ingredient_search_text=search_text,
                instructions=instructions,
                image_name=image_name,
            )
        return recipes
