"""Local hybrid retrieval ablations and inventory-use measurements."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from app.config import Settings
from evaluation.benchmarks import load_json_records
from evaluation.metrics import mrr_at_k, percentile, recall_at_k
from evaluation.models import EvidenceKind, SampleResult, SuiteResult
from recommender.ingredients import contains_blocked_ingredient
from recommender.models import NormalizedRecipeRequest
from recommender.query import build_fts_query, build_semantic_query
from recommender.search import LocalRecipeRecommender, load_recommender, reciprocal_rank_fusion


def inventory_coverage(recipe: dict[str, object]) -> float:
    used = int(recipe.get("used_ingredient_count", 0))
    missing = int(recipe.get("missing_ingredient_count", 0))
    total = used + missing
    return used / total if total else 0.0


def retrieve_ablation(
    recommender: LocalRecipeRecommender,
    request: NormalizedRecipeRequest,
    limit: int = 10,
) -> dict[str, list[int]]:
    fts_ids = recommender.repository.fts_search(
        build_fts_query(request.ingredients), limit=100
    )
    vector = recommender.encoder.encode_query(build_semantic_query(request))
    _, raw_ids = recommender.faiss_index.search(vector, 100)
    semantic_ids = [int(value) for value in raw_ids[0] if int(value) >= 0]
    fused = reciprocal_rank_fusion(fts_ids, semantic_ids)
    rrf_ids = [
        recipe_id
        for recipe_id, _ in sorted(
            fused.items(), key=lambda item: (-item[1].score, item[0])
        )
    ]
    return {
        "fts": fts_ids[:limit],
        "faiss": semantic_ids[:limit],
        "rrf": rrf_ids[:limit],
    }


def _rank(values: list[int], target: int) -> int | None:
    try:
        return values.index(target) + 1
    except ValueError:
        return None


def _request(record: dict[str, object]) -> NormalizedRecipeRequest:
    return NormalizedRecipeRequest(
        ingredients=tuple(str(value) for value in record["ingredients"]),
        exclude_ingredients=frozenset(
            str(value) for value in record.get("exclude_ingredients", [])
        ),
        blocked_ingredients=frozenset(
            str(value) for value in record.get("blocked_ingredients", [])
        ),
    )


def run_retrieval_suite(
    settings: Settings, targets_path: Path, inventory_path: Path
) -> SuiteResult:
    recommender = load_recommender(
        settings.recommender_db_path,
        settings.recommender_faiss_path,
        settings.recommender_metadata_path,
        settings.recommender_model_path,
        settings.recommender_model_revision,
    )
    targets = load_json_records(
        targets_path,
        required_keys={
            "id", "query", "ingredients", "target_recipe_id", "construction_type"
        },
    )
    inventory_records = load_json_records(
        inventory_path,
        required_keys={
            "id", "ingredients", "exclude_ingredients", "blocked_ingredients"
        },
    )
    samples: list[SampleResult] = []
    ranks: dict[str, list[int | None]] = {"fts": [], "faiss": [], "rrf": []}
    unique_fts = 0
    unique_faiss = 0
    for record in targets:
        target = int(record["target_recipe_id"])
        if target not in recommender.repository.get_many([target]):
            samples.append(
                SampleResult(
                    id=str(record["id"]), status="failed",
                    evidence=EvidenceKind.MEASURED,
                    error=f"target recipe {target} is absent",
                )
            )
            continue
        request = _request(record)
        started = perf_counter()
        results = retrieve_ablation(recommender, request)
        latency = (perf_counter() - started) * 1000
        sample_ranks = {name: _rank(values, target) for name, values in results.items()}
        for name, rank in sample_ranks.items():
            ranks[name].append(rank)
        if sample_ranks["fts"] is not None and sample_ranks["faiss"] is None:
            unique_fts += 1
        if sample_ranks["faiss"] is not None and sample_ranks["fts"] is None:
            unique_faiss += 1
        samples.append(
            SampleResult(
                id=str(record["id"]), status="passed",
                evidence=EvidenceKind.MEASURED, latency_ms=[latency],
                metrics={
                    "dataset": "retrieval_target",
                    "target_recipe_id": target,
                    "ranks": sample_ranks,
                    "construction_type": record["construction_type"],
                },
            )
        )

    coverages: list[float] = []
    used_counts: list[int] = []
    missing_counts: list[int] = []
    violations = 0
    returned = 0
    degraded = 0
    inventory_latencies: list[float] = []
    for record in inventory_records:
        request = _request(record)
        recommender.search(request)
        repetitions: list[float] = []
        recipes: list[dict[str, object]] = []
        for _ in range(5):
            started = perf_counter()
            recipes = recommender.search(request)
            repetitions.append((perf_counter() - started) * 1000)
        top = recipes[:3]
        if len(top) < 3:
            degraded += 1
        for recipe in top:
            coverages.append(inventory_coverage(recipe))
            used_counts.append(int(recipe["used_ingredient_count"]))
            missing_counts.append(int(recipe["missing_ingredient_count"]))
            returned += 1
            if contains_blocked_ingredient(
                recipe["ingredients"], request.blocked_ingredients
            ):
                violations += 1
        inventory_latencies.extend(repetitions)
        samples.append(
            SampleResult(
                id=str(record["id"]), status="passed",
                evidence=EvidenceKind.MEASURED, latency_ms=repetitions,
                metrics={
                    "dataset": "inventory_scenario",
                    "result_count": len(recipes),
                    "top3_mean_coverage": (
                        sum(inventory_coverage(recipe) for recipe in top) / len(top)
                        if top else 0.0
                    ),
                },
            )
        )

    best_single_recall = max(recall_at_k(ranks["fts"], 10), recall_at_k(ranks["faiss"], 10))
    rrf_recall = recall_at_k(ranks["rrf"], 10)
    return SuiteResult(
        name="local_retrieval", evidence=EvidenceKind.MEASURED, samples=samples,
        summary={
            "target_count": len(targets),
            "inventory_scenario_count": len(inventory_records),
            "fts_recall_at_3": recall_at_k(ranks["fts"], 3),
            "fts_recall_at_10": recall_at_k(ranks["fts"], 10),
            "fts_mrr_at_10": mrr_at_k(ranks["fts"], 10),
            "faiss_recall_at_3": recall_at_k(ranks["faiss"], 3),
            "faiss_recall_at_10": recall_at_k(ranks["faiss"], 10),
            "faiss_mrr_at_10": mrr_at_k(ranks["faiss"], 10),
            "rrf_recall_at_3": recall_at_k(ranks["rrf"], 3),
            "rrf_recall_at_10": rrf_recall,
            "rrf_mrr_at_10": mrr_at_k(ranks["rrf"], 10),
            "rrf_recall_delta_over_best_single": rrf_recall - best_single_recall,
            "fts_unique_hits_at_10": unique_fts,
            "faiss_unique_hits_at_10": unique_faiss,
            "top3_mean_inventory_coverage": sum(coverages) / len(coverages),
            "top3_mean_used_ingredients": sum(used_counts) / len(used_counts),
            "top3_mean_missing_ingredients": sum(missing_counts) / len(missing_counts),
            "blocked_ingredient_violations": violations,
            "blocked_ingredient_violation_rate": violations / returned if returned else 0.0,
            "degraded_scenario_rate": degraded / len(inventory_records),
            "inventory_latency_p50_ms": percentile(inventory_latencies, 0.5),
            "inventory_latency_p95_ms": percentile(inventory_latencies, 0.95),
        },
    )
