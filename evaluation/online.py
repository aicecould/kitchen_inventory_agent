"""Paired real-model cost probes for requests blocked by the domain gate."""

from __future__ import annotations

from time import perf_counter

from langchain_openai import ChatOpenAI

from app.llm_log import callback_store, logged_span, model_callbacks
from app.config import Settings
from evaluation.models import EvidenceKind, SampleResult, SuiteResult
from evaluation.usage import UsageCollector


def run_gate_cost_probe(
    settings: Settings, samples: list[str], *, offline: bool
) -> SuiteResult:
    if offline:
        return SuiteResult(
            name="gate_cost_probe", evidence=EvidenceKind.SKIPPED,
            samples=[
                SampleResult(
                    id=f"gate-cost-{index:02d}", status="skipped",
                    evidence=EvidenceKind.SKIPPED,
                    metrics={"text": text, "reason": "offline_mode"},
                )
                for index, text in enumerate(samples, start=1)
            ],
            summary={"sample_count": len(samples), "online_skipped": len(samples)},
        )
    logging_callbacks = model_callbacks(settings, "online_evaluation")
    model = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
        timeout=settings.http_timeout_seconds,
        max_retries=1,
        max_tokens=min(settings.deepseek_max_output_tokens, 200),
    )
    results: list[SampleResult] = []
    for index, text in enumerate(samples, start=1):
        collector = UsageCollector()
        started = perf_counter()
        try:
            with logged_span(
                callback_store(logging_callbacks), "request", "gate_cost_probe", prompt=text,
                metadata={"source": "evaluation", "suite": "gate_cost_probe",
                          "sample_id": f"gate-cost-{index:02d}"},
            ) as query:
                response = model.invoke(
                    [
                        {"role": "system", "content": (
                            "You are a kitchen inventory assistant. Briefly state that this "
                            "request is outside kitchen inventory and cooking scope."
                        )},
                        {"role": "user", "content": text},
                    ],
                    config={"callbacks": [collector, *logging_callbacks], "metadata": {
                        "log_component": "gate_cost_probe",
                        "log_metadata": {"source": "evaluation", "suite": "gate_cost_probe",
                                         "sample_id": f"gate-cost-{index:02d}"},
                    }},
                )
                query["content"] = response
            status = "passed"
            error = None
        except Exception as exc:
            status = "failed"
            error = type(exc).__name__
        usage = collector.snapshot()
        results.append(
            SampleResult(
                id=f"gate-cost-{index:02d}", status=status,
                evidence=EvidenceKind.PAIRED_ESTIMATE,
                latency_ms=[(perf_counter() - started) * 1000],
                metrics={
                    "text": text,
                    "gate_model_tokens": 0,
                    "bypassed_model_tokens": usage.total_tokens if usage else None,
                    "saved_tokens": usage.total_tokens if usage else None,
                },
                usage=usage,
                call_counts={"deepseek": collector.call_count},
                error=error,
            )
        )
    available = [
        sample.usage.total_tokens
        for sample in results
        if sample.usage is not None and sample.usage.total_tokens is not None
    ]
    return SuiteResult(
        name="gate_cost_probe", evidence=EvidenceKind.PAIRED_ESTIMATE,
        samples=results,
        summary={
            "sample_count": len(results),
            "successful_calls": sum(sample.status == "passed" for sample in results),
            "average_tokens_saved_per_block": (
                sum(available) / len(available) if available else None
            ),
            "usage_available_count": len(available),
        },
    )
