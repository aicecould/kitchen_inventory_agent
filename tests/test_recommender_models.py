from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import get_settings
from recommender.models import IngredientTranslation, RecipeRanking


def test_recommender_and_agent_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEEPSEEK_MAX_INPUT_CHARS",
        "RECOMMENDER_ENABLED",
        "RECOMMENDER_DB_PATH",
        "RECOMMENDER_FAISS_PATH",
        "RECOMMENDER_METADATA_PATH",
        "RECOMMENDER_MODEL_PATH",
        "RECOMMENDER_MODEL_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.deepseek_max_input_chars == 60000
    assert settings.recommender_enabled is True
    assert settings.recommender_db_path == Path("recommender/artifacts/recipes.db")
    assert settings.recommender_faiss_path == Path("recommender/artifacts/recipes.faiss")
    assert settings.recommender_metadata_path == Path(
        "recommender/artifacts/metadata.json"
    )
    assert settings.recommender_model_path == Path(".run/models/bge-m3")
    assert settings.recommender_model_revision == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_recipe_ranking_values_are_stable() -> None:
    assert [item.value for item in RecipeRanking] == [
        "default",
        "max-used-ingredients",
        "min-missing-ingredients",
    ]


def test_translation_strips_values_and_forbids_extra_fields() -> None:
    value = IngredientTranslation(original=" 西红柿 ", api_term=" tomato ")
    assert value.model_dump() == {"original": "西红柿", "api_term": "tomato"}
    with pytest.raises(ValidationError):
        IngredientTranslation.model_validate(
            {"original": "番茄", "api_term": "tomato", "lang": "zh"}
        )
