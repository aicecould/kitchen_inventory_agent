from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import get_settings
from recommender.models import BuildMetadata, IngredientTranslation, RecipeRanking


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
    assert settings.recommender_model_path == Path("recommender/model")
    assert settings.recommender_model_revision == (
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )


def test_metadata_contract_uses_bge_small() -> None:
    metadata = BuildMetadata(
        source_commit="c5a1e8dab69d46841d914be1f20cca6fbe4b6260",
        source_sha256="a" * 64,
        sqlite_sha256="b" * 64,
        license="CC BY-SA 3.0",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        embedding_max_seq_length=512,
        embedding_dimension=384,
        recipe_count=2,
        built_at=datetime.now(UTC),
    )

    assert metadata.embedding_model == "BAAI/bge-small-en-v1.5"
    assert metadata.embedding_model_revision == (
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )
    assert metadata.embedding_max_seq_length == 512


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
