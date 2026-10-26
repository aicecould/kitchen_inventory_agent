from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from classifier.model import ClassificationResult
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import Field
from PIL import Image

from app.adapters.vision_api import Detection
from app.adapters.recipe_api import TheMealDbClient
from app.agent import KitchenAgent
from app.audit import AuditCode, AuditResult
from app.config import get_settings
from app.context import AgentContext, AgentResult
from app.memory import ConversationOutcome
from app.profile_actions import ProfileChangeRequest
from app.pipeline import build_pipeline
from app.tools.recipe import RecipeRouter
from app.supervisor import RepairAction, RepairDecision


class ForbiddenClassifier:
    calls = 0

    def classify(self, text: str) -> ClassificationResult:
        self.calls += 1
        raise AssertionError(f"classifier must not receive deterministic route: {text}")


class IrrelevantClassifierSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> ClassificationResult:
        self.calls.append(text)
        return ClassificationResult(
            is_related=False, positive_score=0.01, threshold=0.5
        )


class RelatedClassifierSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> ClassificationResult:
        self.calls.append(text)
        return ClassificationResult(
            is_related=True, positive_score=0.99, threshold=0.5
        )


class CapturingAgent:
    def __init__(self) -> None:
        self.contexts: list[AgentContext] = []

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        self.contexts.append(context)
        return AgentResult(content="已根据上文继续处理。")


class RecipeResultAgent(CapturingAgent):
    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        self.contexts.append(context)
        return AgentResult(
            content="推荐 Tomato Soup：https://recipes.example/tomato-soup",
            tool_history=[
                {
                    "tool": "search_recipes",
                    "arguments": {"ingredients": ["tomato"]},
                    "result": [
                        {
                            "id": "recipe-1",
                            "title": "Tomato Soup",
                            "instructions": "Simmer.",
                            "ingredients": ["tomato"],
                            "source": "example",
                            "source_url": "https://recipes.example/tomato-soup",
                            "image_url": None,
                        }
                    ],
                }
            ],
        )


class ReconciliationProposalAgent(CapturingAgent):
    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        self.contexts.append(context)
        return AgentResult(
            content="已生成待确认操作，确认前库存不会改变。",
            tool_history=[
                {
                    "tool": "propose_remove_inventory_instance",
                    "arguments": {"instance_id": "inv_" + "1" * 32},
                    "result": {
                        "action_id": "act_" + "1" * 32,
                        "operation": "inventory.remove",
                        "status": "pending",
                        "source_task_id": self.task_id,
                    },
                }
            ],
        )


class SummarySpy:
    def __init__(self) -> None:
        self.users: list[str] = []

    def run_if_due(self, user_id: str) -> list[object]:
        self.users.append(user_id)
        return []


class FixedExpiryInferer:
    def infer(self, name: str, now: datetime) -> datetime:
        assert name == "番茄"
        return datetime(2099, 1, 2, tzinfo=UTC)


class ScriptedChatModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)
    bound_tool_names_by_call: list[list[str]] = Field(default_factory=list)
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        self.bound_tool_names = [
            str(getattr(tool, "name", tool.get("name") if isinstance(tool, dict) else ""))
            for tool in tools
        ]
        self.bound_tool_names_by_call.append(list(self.bound_tool_names))
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
        self.received_messages.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class FailingChatModel(ScriptedChatModel):
    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
        raise RuntimeError("provider unavailable")


def build_test_pipeline(tmp_path: Path):
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    return build_pipeline(settings)


def test_pipeline_reserves_segment_id_before_agent_and_persists_same_id(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    agent = CapturingAgent()
    pipeline.agent = agent  # type: ignore[assignment]
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]

    pipeline.process_request(user_id="u", text="土豆怎么做")

    record = pipeline.history.list_records("u")[-1]
    assert agent.contexts[0].segment_id == record.segment_id


@pytest.mark.parametrize("text", [
    "根据现有库存推荐一道菜", "根据现有库存推荐一道菜，并避开我的过敏原",
])
def test_inventory_recipe_request_bypasses_classifier_and_keeps_recipe_intent(tmp_path, text):
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier
    agent = CapturingAgent()
    pipeline.agent = agent
    pipeline.process_request(user_id="u", text=text)
    assert classifier.calls == 0
    assert agent.contexts[0].intent == "recipe"


def test_pipeline_injects_active_task_without_partitioning_global_history(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    pipeline.history.append(user_id="u", request="global", response="history")
    evidence = pipeline.history.append(
        user_id="u",
        request="用番茄找菜谱",
        response="找到番茄汤",
        tool_history=[
            {
                "tool": "search_recipes",
                "arguments": {},
                "result": [FixedRecipeRouter().search()[0]],
            }
        ],
    )
    pipeline.tasks.confirm_recipe(
        user_id="u",
        segment_id=evidence.segment_id,
        recipe_id="recipe-1",
        servings=2,
        default_context=[],
        as_new=False,
        expected_version=None,
        request_id="confirm-1",
    )
    agent = CapturingAgent()
    pipeline.agent = agent  # type: ignore[assignment]
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]

    pipeline.process_request(user_id="u", text="接着说")

    context = agent.contexts[0]
    assert [item.request for item in context.history_segments] == [
        "global",
        "用番茄找菜谱",
    ]
    assert context.cooking_task is not None
    assert context.cooking_task.recipe_title == "Tomato Soup"
    assert context.cooking_task.servings == 2


def test_pipeline_returns_server_derived_recipe_options_and_segment_id(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    agent = RecipeResultAgent()
    pipeline.agent = agent  # type: ignore[assignment]
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]

    result = pipeline.process_request(user_id="u", text="找番茄汤")

    assert result.segment_id.startswith("hist_")
    assert result.recipe_options[0]["id"] == "recipe-1"
    persisted = pipeline.history.get_evidence("u", [result.segment_id])[0]
    assert persisted.tool_history[-1]["tool"] == "search_recipes"


def test_reconciliation_request_binds_completed_task_and_closes_after_proposal(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    evidence = pipeline.history.append(
        user_id="u",
        request="用番茄找菜谱",
        response="找到番茄汤",
        tool_history=[
            {
                "tool": "search_recipes",
                "arguments": {},
                "result": [FixedRecipeRouter().search()[0]],
            }
        ],
    )
    active = pipeline.tasks.confirm_recipe(
        user_id="u",
        segment_id=evidence.segment_id,
        recipe_id="recipe-1",
        servings=1,
        default_context=[],
        as_new=False,
        expected_version=None,
        request_id="confirm-1",
    )
    completed, _ = pipeline.tasks.complete_cooked(
        user_id="u",
        task_id=active.task_id,
        expected_version=active.version,
        request_id="cooked-1",
    )
    agent = ReconciliationProposalAgent(completed.task_id)
    pipeline.agent = agent  # type: ignore[assignment]

    pipeline.process_request(
        user_id="u",
        text="番茄已经用完",
        reconciliation_task_id=completed.task_id,
    )

    context = agent.contexts[0]
    assert context.inventory_source_task_id == completed.task_id
    assert context.cooking_task is not None
    assert context.cooking_task.task_id == completed.task_id
    assert pipeline.tasks.repository.get(completed.task_id).reconciliation_open is False  # type: ignore[union-attr]


def test_reconciliation_rejects_active_or_cross_user_task(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    evidence = pipeline.history.append(
        user_id="u",
        request="用番茄找菜谱",
        response="找到番茄汤",
        tool_history=[
            {
                "tool": "search_recipes",
                "arguments": {},
                "result": [FixedRecipeRouter().search()[0]],
            }
        ],
    )
    active = pipeline.tasks.confirm_recipe(
        user_id="u",
        segment_id=evidence.segment_id,
        recipe_id="recipe-1",
        servings=1,
        default_context=[],
        as_new=False,
        expected_version=None,
        request_id="confirm-1",
    )

    with pytest.raises(ValueError, match="reconciliation"):
        pipeline.process_request(
            user_id="u", text="核对", reconciliation_task_id=active.task_id
        )
    with pytest.raises(PermissionError, match="another user"):
        pipeline.process_request(
            user_id="other", text="核对", reconciliation_task_id=active.task_id
        )


def test_pipeline_checks_profile_summary_after_persisting_result(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    summarizer = SummarySpy()
    pipeline.profile_summarizer = summarizer  # type: ignore[attr-defined]

    pipeline.process_request(user_id="u", text="查看库存")

    assert summarizer.users == ["u"]
    assert len(pipeline.history.list_records("u")) == 1


def test_pipeline_resolves_profile_proposal_and_reloads_active_profile(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    evidence = pipeline.history.append(
        user_id="u", request="以后不吃香菜", response="已生成提案"
    )
    proposal = pipeline.profile_actions.propose(
        "u",
        ProfileChangeRequest(
            operation="upsert",
            category="preference",
            key="ingredient.cilantro",
            value="不喜欢香菜",
            reason="用户明确表达永久偏好",
        ),
        [evidence.segment_id],
    )

    resolved = pipeline.resolve_profile_proposal(
        user_id="u", proposal_id=proposal.proposal_id, decision="confirm"
    )

    assert resolved.status == "confirmed"
    assert pipeline.profile.preferences == ["不喜欢香菜"]
    confirmation = pipeline.history.list_records("u")[-1]
    assert confirmation.outcome_type == ConversationOutcome.CONFIRMATION
    assert confirmation.confirmation_result == {
        "type": "profile",
        "decision": "confirm",
        "proposal_id": proposal.proposal_id,
        "status": "confirmed",
    }


def test_pipeline_enables_deepseek_web_search_fallback_with_api_key(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )

    pipeline = build_pipeline(settings)

    assert pipeline.agent is not None
    assert pipeline.agent.recipes.web_search is not None
    assert (
        pipeline.agent.recipes.web_search.endpoint
        == "https://api.deepseek.com/anthropic/v1/messages"
    )


def install_scripted_agent(
    pipeline: Any,
    responses: list[AIMessage],
    recipes: Any | None = None,
    repair_decider: Any | None = None,
) -> ScriptedChatModel:
    model = ScriptedChatModel(responses=responses)
    recipe_router = recipes or RecipeRouter(
        None,
        None,
        TheMealDbClient(
            base_url=pipeline.settings.themealdb_base_url,
            api_key=pipeline.settings.themealdb_api_key,
            timeout=pipeline.settings.http_timeout_seconds,
        ),
    )
    pipeline.agent = KitchenAgent(
        model=model,  # type: ignore[arg-type]
        inventory=pipeline.inventory,
        recipes=recipe_router,
        actions=pipeline.actions,
        repair_decider=repair_decider,
    )
    return model


def test_recipe_tool_errors_can_be_corrected_without_becoming_recipe_evidence(tmp_path):
    pipeline = build_test_pipeline(tmp_path)
    pipeline.inventory.add_instance("番茄", "2个", None, "unknown")
    model = install_scripted_agent(pipeline, [
        AIMessage(content="", tool_calls=[{"name": "search_recipes", "id": "bad",
                  "args": {"ingredients": [], "ingredient_mapping": []}}]),
        AIMessage(content="", tool_calls=[{"name": "search_recipes", "id": "good",
                  "args": {"ingredients": ["tomato"], "ingredient_mapping": []}}]),
        AIMessage(content="推荐 Tomato Soup（番茄汤），将番茄煮软。"),
    ], recipes=FixedRecipeRouter())
    result = pipeline.process_request(user_id="u", text="根据库存推荐一道菜")
    assert result.public_payload()["state"] == "completed"
    assert result.tool_history[0]["origin"] == "business_rule"
    assert result.tool_history[1]["status"] == "error"
    assert result.tool_history[2]["result"][0]["title"] == "Tomato Soup"
    assert all(names == ["search_recipes"] for names in model.bound_tool_names_by_call)
    assert pipeline.actions.repository.list_pending("u") == []


class FixedRecipeRouter:
    def search(self, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "id": "recipe-1",
                "title": "Tomato Soup",
                "instructions": "Simmer.",
                "ingredients": ["tomato"],
                "source": "example",
                "source_url": "https://recipes.example/tomato-soup",
                "image_url": None,
            }
        ]


def test_repair_subagent_rewrites_audit_only_failure_once(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]
    repair_calls: list[AuditCode] = []

    def repair_decider(request: object, audit: AuditResult) -> RepairDecision:
        repair_calls.append(audit.code)
        return RepairDecision(
            action=RepairAction.REWRITE,
            content="当前不支持下单。",
            reason_code=AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        )

    install_scripted_agent(
        pipeline,
        [AIMessage(content="已经完成下单")],
        repair_decider=repair_decider,
    )

    result = pipeline.process_request(user_id="test", text="帮我处理一下")

    assert result.content == "当前不支持下单。"
    assert result.blocked is False
    assert repair_calls == [AuditCode.REPAIRABLE_OUTPUT_VIOLATION]
    assert [
        event.name
        for event in result.execution_trace
        if event.stage == "supervisor"
    ] == [
        "response_audit",
        "supervisor_classification",
        "subagent_rewrite",
        "subagent_reaudit",
    ]


def test_failed_subagent_rewrite_uses_one_toolless_main_retry(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]

    def repair_decider(request: object, audit: AuditResult) -> RepairDecision:
        return RepairDecision(
            action=RepairAction.REWRITE,
            content="已经完成下单",
            reason_code=AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        )

    model = install_scripted_agent(
        pipeline,
        [
            AIMessage(content="已经完成下单"),
            AIMessage(content="当前不支持订单操作。"),
        ],
        repair_decider=repair_decider,
    )

    result = pipeline.process_request(user_id="test", text="帮我处理一下")

    assert result.content == "当前不支持订单操作。"
    assert result.blocked is False
    assert len(model.bound_tool_names_by_call) == 1
    assert pipeline.actions.repository.list_pending("test") == []
    assert len(pipeline.history.list_records("test")) == 1


def test_missing_recipe_tool_retries_main_with_search_recipes_only(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]

    def repair_decider(request: object, audit: AuditResult) -> RepairDecision:
        return RepairDecision(
            action=RepairAction.RETRY_MAIN,
            feedback="必须先调用 search_recipes。",
            reason_code=AuditCode.RECIPE_TOOL_MISSING,
        )

    model = install_scripted_agent(
        pipeline,
        [
            AIMessage(content="推荐一道番茄汤菜谱，步骤是煮熟番茄。"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_recipes",
                        "args": {
                            "ingredients": ["tomato"],
                            "ingredient_mapping": [],
                            "exclude_ingredients": [],
                        },
                        "id": "call-recipe-retry",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="推荐 Tomato Soup 菜谱，步骤是 Simmer。"),
        ],
        recipes=FixedRecipeRouter(),
        repair_decider=repair_decider,
    )

    result = pipeline.process_request(user_id="test", text="推荐一道番茄菜谱")

    assert result.blocked is False
    assert "Tomato Soup" in result.content
    assert model.bound_tool_names_by_call[-1] == ["search_recipes"]
    assert [entry["tool"] for entry in result.tool_history] == ["search_recipes"]


def test_initial_agent_failure_returns_fixed_task_impossible_response(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]
    pipeline.agent = KitchenAgent(
        model=FailingChatModel(responses=[]),  # type: ignore[arg-type]
        inventory=pipeline.inventory,
        recipes=FixedRecipeRouter(),  # type: ignore[arg-type]
        actions=pipeline.actions,
    )

    result = pipeline.process_request(user_id="test", text="推荐一道番茄菜谱")

    assert result.content == "当前无法完成该请求，库存未发生变化。"
    assert result.blocked is False
    assert result.audit_reason == AuditCode.TASK_IMPOSSIBLE.value
    assert [event.name for event in result.execution_trace[-2:]] == [
        "deepseek_agent",
        "final_fallback",
    ]


def test_non_food_query_is_blocked_before_deepseek(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text(
        "# 用户画像\n\n## 过敏原\n\n- 暂无\n",
        encoding="utf-8",
    )
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )

    pipeline = build_pipeline(settings)
    result = pipeline.process_request(user_id="test", text="Python怎么安装")

    assert result.blocked is True
    assert result.audit_reason == "food_query_irrelevant"
    assert "仅支持食谱和食材" in result.content
    records = pipeline.history.list_records("test")
    assert len(records) == 1
    assert records[0].response == result.content
    assert records[0].is_relevant is False
    assert records[0].outcome_type == ConversationOutcome.BLOCKED
    assert records[0].blocked is True
    assert records[0].execution_trace == result.execution_trace
    assert pipeline.history.list_context_window("test") == []


class LowConfidenceVision:
    def recognize(self, image_bytes: bytes) -> list[Detection]:
        assert image_bytes
        return [Detection(name="土豆", confidence=0.2)]


def _valid_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_waiting_image_confirmation_and_resolution_are_distinct_records(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    pipeline.vision = LowConfidenceVision()  # type: ignore[assignment]
    pipeline.agent = CapturingAgent()  # type: ignore[assignment]

    waiting = pipeline.process_request(
        user_id="test",
        text="识别后推荐菜谱",
        image_bytes=_valid_png(),
    )

    assert waiting.awaiting_vision_confirmation is True
    confirmation_id = waiting.vision_confirmation.confirmation_id  # type: ignore[union-attr]
    first = pipeline.history.list_records("test")
    assert len(first) == 1
    assert first[0].outcome_type == ConversationOutcome.AWAITING_CONFIRMATION
    assert first[0].request == "识别后推荐菜谱"

    resolved = pipeline.resolve_vision_confirmation(
        user_id="test",
        confirmation_id=confirmation_id,
        selections=[],
    )

    assert resolved.content == "已根据上文继续处理。"
    records = pipeline.history.list_records("test")
    assert len(records) == 2
    assert records[1].outcome_type == ConversationOutcome.CONFIRMATION
    assert records[1].request == f"图片确认：{confirmation_id}"
    assert records[1].confirmation_result == {
        "type": "vision",
        "confirmation_id": confirmation_id,
        "selected": [],
        "dropped": [
            {
                "candidate_index": 0,
                "name": "土豆",
                "confidence": 0.2,
            }
        ],
    }


def test_inventory_confirmation_is_recorded_as_confirmation(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    action = pipeline.actions.propose_add("test", "牛奶", "一盒")
    executed = pipeline.actions.confirm(action.action_id, "test")

    pipeline.record_inventory_confirmation(action=executed, decision="confirm")

    records = pipeline.history.list_records("test")
    assert len(records) == 1
    assert records[0].outcome_type == ConversationOutcome.CONFIRMATION
    assert records[0].is_relevant is True
    assert records[0].confirmation_result["decision"] == "confirm"  # type: ignore[index]


def test_simple_write_reaches_agent_and_preserves_quantity_text(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    pipeline = build_pipeline(settings)
    pipeline.actions.expiry_inferer = FixedExpiryInferer()
    classifier = RelatedClassifierSpy()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_add_inventory_instance",
                        "args": {"name": "番茄", "quantity_text": "02个"},
                        "id": "call-add",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成待确认操作，确认前库存不会改变。"),
        ],
    )

    result = pipeline.process_request(user_id="test", text="添加 02个 番茄到库存")

    assert classifier.calls == ["添加 02个 番茄到库存"]
    assert "待确认" in result.content
    assert pipeline.inventory.find_by_name("番茄") == []
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].arguments == {
        "name": "番茄",
        "quantity_text": "02个",
        "expires_at": "2099-01-02T00:00:00Z",
        "expiry_source": "llm",
    }
    assert result.tool_history[0]["tool"] == "propose_add_inventory_instance"
    assert all(
        event.name != "expiry_inference" for event in result.execution_trace
    )


def test_agent_add_records_safe_unknown_expiry_degradation(tmp_path: Path) -> None:
    """A no-provider add must disclose unknown inference without provider details."""
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_add_inventory_instance",
                        "args": {"name": "土豆", "quantity_text": "2个"},
                        "id": "call-add",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成待确认操作，确认前库存不会改变。"),
        ],
    )

    result = pipeline.process_request(user_id="test", text="添加2个土豆到库存")

    events = [
        event for event in result.execution_trace if event.name == "expiry_inference"
    ]
    assert len(events) == 1
    assert events[0].stage == "model"
    assert events[0].status == "degraded"
    assert events[0].detail == "unknown expiry; proposal remains pending"


def test_missing_classifier_artifact_fails_open(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
        food_classifier_model_path=tmp_path / "missing.bin",
        food_classifier_metadata_path=tmp_path / "missing.json",
    )
    pipeline = build_pipeline(settings)

    result = pipeline.process_request(user_id="test", text="帮我加入购物车")

    assert result.blocked is False
    classifier_event = result.execution_trace[1]
    assert classifier_event.name == "food_query_classifier"
    assert classifier_event.status == "unavailable"
    assert "fail-open" in classifier_event.detail


def test_inventory_query_returns_exact_instance_snapshots_without_deepseek(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    pipeline = build_pipeline(settings)
    item = pipeline.inventory.add_instance(
        "土豆", "2 个", None, "unknown", now=datetime(2026, 8, 7, tzinfo=UTC)
    )

    result = pipeline.process_request(user_id="test", text="查询一下当前库存")

    assert result.content == (
        "当前库存：\n- 土豆：2 个；过期 未知"
    )
    assert result.blocked is False
    assert result.tool_history == [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [item],
        }
    ]
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route",
        "intent_match",
        "inventory_query_regex",
        "list_inventory",
        "regex_audit",
        "tool_call_audit",
    ]
    assert all(event.duration_ms >= 0 for event in result.execution_trace)


def test_irrelevant_greeting_bypasses_classifier_with_boundary_result(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="你好")

    assert result.blocked is True
    assert result.audit_reason == "food_query_irrelevant"
    assert result.content == "当前仅支持食谱和食材相关问题。"
    assert classifier.calls == 0
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route"
    ]
    assert result.execution_trace[0].detail == "irrelevant"


def test_explicit_text_cancellation_bypasses_models_with_fixed_response(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="不要了")

    assert result.content == "已停止处理；没有执行新的库存变更。"
    assert result.blocked is False
    assert result.audit_reason == AuditCode.USER_CANCELLED.value
    assert pipeline.history.list_records("test")[0].audit_reason == (
        AuditCode.USER_CANCELLED.value
    )
    assert classifier.calls == 0
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route",
        "intent_match",
        "request_cancel",
        "regex_audit",
        "tool_call_audit",
    ]


def test_supported_list_bypasses_classifier_and_returns_real_inventory(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    item = pipeline.inventory.add_instance("土豆", "大约100个", None, "unknown")

    result = pipeline.process_request(user_id="test", text="查询库存")

    assert result.content == (
        "当前库存：\n- 土豆：大约100个；过期 未知"
    )
    assert classifier.calls == 0
    assert result.execution_trace[0].name == "deterministic_route"
    assert result.execution_trace[0].detail == "supported; inventory.list"


def test_supported_named_read_bypasses_classifier_and_returns_exact_instances(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    first = pipeline.inventory.add_instance("土豆", "2个", None, "unknown")
    second = pipeline.inventory.add_instance("土豆", "半袋", None, "unknown")

    result = pipeline.process_request(user_id="test", text="查看土豆还有多少")

    assert result.content == (
        "当前库存：\n- 土豆：2个；过期 未知\n- 土豆：半袋；过期 未知"
    )
    assert result.tool_history == [
        {
            "tool": "find_inventory_instances",
            "arguments": {"name": "土豆"},
            "result": [first, second],
        }
    ]
    assert classifier.calls == 0
    assert [event.name for event in result.execution_trace[:3]] == [
        "deterministic_route",
        "intent_match",
        "inventory_query_regex",
    ]


def test_supported_remove_all_bypasses_classifier_and_stays_pending(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    pipeline.inventory.add_instance("土豆", "2个", None, "unknown")
    pipeline.inventory.add_instance("土豆", "1袋", None, "unknown")

    result = pipeline.process_request(user_id="test", text="删除所有土豆")

    assert "待确认" in result.content
    assert len(pipeline.inventory.find_by_name("土豆")) == 2
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.remove_all"
    assert pending[0].arguments == {"name": "土豆"}
    assert result.tool_history[0]["tool"] == "propose_remove_all_inventory_instances"
    assert classifier.calls == 0
    assert result.execution_trace[0].detail == "supported; inventory.remove_all"


def test_shared_prefix_instances_are_rendered_with_distinct_full_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two colliding decorative prefixes must remain separately selectable."""
    values = iter(
        [
            "abcd1111111111111111111111111111",
            "abcd2222222222222222222222222222",
        ]
    )
    monkeypatch.setattr(
        "app.tools.inventory.uuid4", lambda: SimpleNamespace(hex=next(values))
    )
    pipeline = build_test_pipeline(tmp_path)
    first = pipeline.inventory.add_instance("番茄", "2个", None, "unknown")
    second = pipeline.inventory.add_instance("番茄", "半盒", None, "unknown")

    result = pipeline.process_request(user_id="test", text="查看番茄还有多少")

    assert str(first["instance_id"]) not in result.content
    assert str(second["instance_id"]) not in result.content
    assert result.content.count("- 番茄：") == 2
    assert result.tool_history[0]["result"] == [first, second]


def test_explicit_full_id_replace_bypasses_classifier_and_stays_pending(
    tmp_path: Path,
) -> None:
    """Opaque replacement text must reach a pending proposal without model routing."""
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    original = pipeline.inventory.add_instance(
        "牛奶",
        "一盒",
        datetime(2026, 8, 20, 9, tzinfo=UTC),
        "user",
    )
    instance_id = str(original["instance_id"])

    result = pipeline.process_request(
        user_id="test",
        text=f"将 {instance_id} 更新为 大约半盒（已开封）",
    )

    assert classifier.calls == 0
    assert result.blocked is False
    assert instance_id not in result.content
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.replace"
    assert pending[0].arguments == {
        "instance_id": instance_id,
        "quantity_text": "大约半盒（已开封）",
        "expires_at": "2026-08-20T09:00:00Z",
        "expiry_source": "user",
    }
    assert pending[0].status == "pending"


@pytest.mark.parametrize(
    ("verb", "operation"),
    [
        ("删除", "inventory.remove"),
        ("确认", "inventory.confirm"),
    ],
)
def test_explicit_full_id_remove_and_confirm_bypass_classifier_and_stay_pending(
    tmp_path: Path, verb: str, operation: str
) -> None:
    """Explicit delete and calibration routes must stop at the confirmation gate."""
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    original = pipeline.inventory.add_instance("鸡蛋", "6个", None, "unknown")
    instance_id = str(original["instance_id"])

    result = pipeline.process_request(user_id="test", text=f"{verb} {instance_id}")

    assert classifier.calls == 0
    assert result.blocked is False
    assert instance_id not in result.content
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == operation
    assert pending[0].arguments == {"instance_id": instance_id}
    assert pending[0].status == "pending"


def test_contextual_route_bypasses_classifier_and_supplies_recent_history(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    agent = CapturingAgent()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    pipeline.agent = agent  # type: ignore[assignment]
    for index in range(14):
        pipeline.history.append(
            user_id="test",
            request=f"土豆步骤 {index}",
            response=f"回答 {index}",
            is_relevant=index not in {11, 13},
            audit_reason=(
                "food_query_irrelevant" if index in {11, 13} else None
            ),
            created_at=datetime(2026, 8, 10, 0, index, tzinfo=UTC),
        )

    result = pipeline.process_request(user_id="test", text="下一步呢")

    assert result.content == "已根据上文继续处理。"
    assert classifier.calls == 0
    assert len(agent.contexts) == 1
    assert [segment.request for segment in agent.contexts[0].history_segments] == [
        "土豆步骤 2",
        "土豆步骤 3",
        "土豆步骤 4",
        "土豆步骤 5",
        "土豆步骤 6",
        "土豆步骤 7",
        "土豆步骤 8",
        "土豆步骤 9",
        "土豆步骤 10",
        "土豆步骤 12",
    ]
    assert all(
        set(segment.model_dump()) == {"request", "response"}
        for segment in agent.contexts[0].history_segments
    )
    assert result.execution_trace[0].name == "deterministic_route"
    assert result.execution_trace[0].detail == "contextual"


def test_relative_use_runs_real_agent_and_keeps_replacement_pending(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    original = pipeline.inventory.add_instance("番茄", "2个", None, "unknown")
    instance_id = str(original["instance_id"])
    model = install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_inventory_instances",
                        "args": {"name": "番茄"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_replace_inventory_instance",
                        "args": {
                            "instance_id": instance_id,
                            "quantity_text": "1个",
                            "expires_at": None,
                            "expiry_source": "unknown",
                        },
                        "id": "call-replace",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成替换库存实例的待确认操作，确认前库存不会改变。"),
        ],
    )
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="用了一半")

    assert result.blocked is False
    assert classifier.calls == 0
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.replace"
    assert pending[0].arguments == {
        "instance_id": instance_id,
        "quantity_text": "1个",
        "expires_at": None,
        "expiry_source": "unknown",
    }
    assert [entry["tool"] for entry in result.tool_history] == [
        "find_inventory_instances",
        "propose_replace_inventory_instance",
    ]
    assert "find_inventory_instances" in model.bound_tool_names
    assert "propose_replace_inventory_instance" in model.bound_tool_names


def test_agent_inventory_proposal_replaces_english_execution_prose_with_canonical_pending(
    tmp_path: Path,
) -> None:
    """Model-authored execution prose must never survive a proposal tool call."""
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_add_inventory_instance",
                        "args": {"name": "牛奶", "quantity_text": "一盒"},
                        "id": "call-add",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I added the milk to inventory."),
        ],
    )

    result = pipeline.process_request(user_id="test", text="请把一盒牛奶加入库存")

    assert result.blocked is False
    assert "I added" not in result.content
    assert "待确认操作（尚未执行）" in result.content
    assert "尚未执行" in result.content
    assert "status=pending" not in result.content
    assert "牛奶" in result.content and "一盒" in result.content
    assert pipeline.inventory.find_by_name("牛奶") == []
    events = [
        event for event in result.execution_trace if event.name == "expiry_inference"
    ]
    assert len(events) == 1
    assert events[0].status == "degraded"
    assert events[0].detail == "unknown expiry; proposal remains pending"


def test_agent_mixed_inventory_recipe_response_preserves_model_recommendation(
    tmp_path: Path,
) -> None:
    """Canonical inventory replacement must append real recipe titles and links."""
    pipeline = build_test_pipeline(tmp_path)
    pipeline.food_classifier = RelatedClassifierSpy()  # type: ignore[assignment]
    item = pipeline.inventory.add_instance("番茄", "大约100个", None, "unknown")
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_inventory_instances",
                        "args": {},
                        "id": "call-list",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_recipes",
                        "args": {
                            "ingredients": ["tomato"],
                            "ingredient_mapping": [],
                            "exclude_ingredients": [],
                        },
                        "id": "call-recipe",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I recommend Tomato Soup. Simmer the tomato. [Recipe](https://recipes.example/tomato-soup)"),
        ],
        recipes=FixedRecipeRouter(),
    )

    result = pipeline.process_request(
        user_id="test",
        text="根据库存推荐一道番茄菜谱",
        target_language="en",
    )

    assert result.blocked is False
    assert result.content == "I recommend Tomato Soup. Simmer the tomato. [Recipe](https://recipes.example/tomato-soup)"
    assert str(item["instance_id"]) not in result.content
    assert result.tool_history[0]["result"] == [item]
    assert result.recipe_options[0]["title"] == "Tomato Soup"
    assert "Current inventory:" not in result.content


def test_relative_replacement_accepts_equivalent_expiry_serializations(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    original = pipeline.inventory.add_instance(
        "番茄",
        "2个",
        datetime(2026, 8, 10, 9, tzinfo=UTC),
        "user",
    )
    instance_id = str(original["instance_id"])
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_inventory_instances",
                        "args": {"name": "番茄"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_replace_inventory_instance",
                        "args": {
                            "instance_id": instance_id,
                            "quantity_text": "1个",
                            "expires_at": "2026-08-10T09:00:00+00:00",
                            "expiry_source": "user",
                        },
                        "id": "call-replace",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成待确认操作，确认前库存不会改变。"),
        ],
    )

    result = pipeline.process_request(user_id="test", text="用了一半")

    assert result.blocked is False
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].arguments == {
        "instance_id": instance_id,
        "quantity_text": "1个",
        "expires_at": "2026-08-10T09:00:00Z",
        "expiry_source": "user",
    }


def test_named_replacement_reaches_agent_lookup_instead_of_direct_write(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = RelatedClassifierSpy()
    original = pipeline.inventory.add_instance("番茄", "2个", None, "unknown")
    instance_id = str(original["instance_id"])
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_inventory_instances",
                        "args": {"name": "番茄"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_replace_inventory_instance",
                        "args": {
                            "instance_id": instance_id,
                            "quantity_text": "1个",
                            "expires_at": None,
                            "expiry_source": "unknown",
                        },
                        "id": "call-replace",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成待确认操作，确认前库存不会改变。"),
        ],
    )
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="将番茄更新为1个")

    assert result.blocked is False
    assert classifier.calls == ["将番茄更新为1个"]
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.replace"
    assert [entry["tool"] for entry in result.tool_history] == [
        "find_inventory_instances",
        "propose_replace_inventory_instance",
    ]


def test_undecided_request_invokes_classifier_once(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = IrrelevantClassifierSpy()
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="Python怎么安装")

    assert classifier.calls == ["Python怎么安装"]
    assert result.blocked is True
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route",
        "food_query_classifier",
    ]


def test_build_runtime_recommender_respects_disabled_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pipeline import build_runtime_recommender

    settings = replace(get_settings(), recommender_enabled=False)
    calls: list[object] = []
    monkeypatch.setattr(
        "app.pipeline.load_recommender",
        lambda **kwargs: calls.append(kwargs),
    )

    assert build_runtime_recommender(settings) is None
    assert calls == []


def test_build_runtime_recommender_returns_none_for_missing_artifacts(
    tmp_path: Path,
) -> None:
    from app.pipeline import build_runtime_recommender

    settings = replace(
        get_settings(),
        recommender_enabled=True,
        recommender_db_path=tmp_path / "missing.db",
        recommender_faiss_path=tmp_path / "missing.faiss",
        recommender_metadata_path=tmp_path / "missing.json",
        recommender_model_path=tmp_path / "missing-model",
    )

    assert build_runtime_recommender(settings) is None


def test_build_runtime_recommender_injects_successful_local_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pipeline import build_runtime_recommender

    fake = object()
    settings = replace(get_settings(), recommender_enabled=True)
    monkeypatch.setattr(
        "app.pipeline.load_recommender",
        lambda **kwargs: fake,
    )

    assert build_runtime_recommender(settings) is fake


def test_build_pipeline_injects_local_recommender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pipeline import build_pipeline

    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    fake_local = object()
    monkeypatch.setattr(
        "app.pipeline.build_runtime_recommender",
        lambda settings_arg: fake_local,
    )

    pipeline = build_pipeline(settings)

    assert pipeline.agent is not None
    assert pipeline.agent.recipes.local is fake_local


def test_build_pipeline_records_local_recommender_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pipeline import build_pipeline

    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    monkeypatch.setattr(
        "app.pipeline.build_runtime_recommender",
        lambda settings_arg: object(),
    )

    pipeline = build_pipeline(settings)

    assert pipeline.local_recipe_search_available is True


def test_pipeline_source_never_imports_build_entrypoint() -> None:
    source = Path("app/pipeline.py").read_text(encoding="utf-8")
    assert "recommender.build" not in source
    assert "python -m recommender.build" not in source
