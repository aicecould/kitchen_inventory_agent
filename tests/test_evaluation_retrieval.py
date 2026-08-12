from dataclasses import replace
from pathlib import Path

import pytest

from app.config import get_settings
from evaluation.retrieval import (
    inventory_coverage,
    retrieve_ablation,
    run_retrieval_suite,
)
from recommender.models import NormalizedRecipeRequest
from recommender.search import load_recommender


ROOT = Path("/home/ubuntu/Education/kitchen_inventory_agent")


@pytest.fixture(scope="module")
def loaded_recommender():
    settings = get_settings()
    return load_recommender(
        ROOT / settings.recommender_db_path,
        ROOT / settings.recommender_faiss_path,
        ROOT / settings.recommender_metadata_path,
        ROOT / settings.recommender_model_path,
        settings.recommender_model_revision,
    )


def test_inventory_coverage_uses_existing_counts() -> None:
    assert inventory_coverage(
        {"used_ingredient_count": 3, "missing_ingredient_count": 1}
    ) == 0.75
    assert inventory_coverage(
        {"used_ingredient_count": 0, "missing_ingredient_count": 0}
    ) == 0.0


def test_ablation_returns_independent_rankings(loaded_recommender) -> None:
    request = NormalizedRecipeRequest(ingredients=("chicken", "rice"))
    result = retrieve_ablation(loaded_recommender, request)
    assert set(result) == {"fts", "faiss", "rrf"}
    assert all(len(values) <= 10 for values in result.values())
    assert all(all(isinstance(value, int) for value in values) for values in result.values())


def test_retrieval_benchmarks_are_fixed_and_targets_exist(loaded_recommender) -> None:
    settings = get_settings()
    absolute = replace(
        settings,
        recommender_db_path=ROOT / settings.recommender_db_path,
        recommender_faiss_path=ROOT / settings.recommender_faiss_path,
        recommender_metadata_path=ROOT / settings.recommender_metadata_path,
        recommender_model_path=ROOT / settings.recommender_model_path,
    )
    result = run_retrieval_suite(
        absolute,
        Path("evaluation/benchmarks/retrieval_targets.json"),
        Path("evaluation/benchmarks/inventory_scenarios.json"),
    )
    assert result.summary["target_count"] == 30
    assert result.summary["inventory_scenario_count"] == 10
    assert result.summary["rrf_recall_at_10"] >= 0
    assert all(sample.error is None for sample in result.samples)
