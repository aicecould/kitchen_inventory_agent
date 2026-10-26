from types import SimpleNamespace

from app.agent import KitchenAgent
from app.audit import AuditCode, AuditResult
from app.supervisor import RepairAction, RepairDecision, SupervisorRequest


class FakeStructuredModel:
    def __init__(self, result: RepairDecision) -> None:
        self.result = result
        self.structured_method: str | None = None

    def with_structured_output(
        self, schema: object, *, method: str
    ) -> "FakeStructuredModel":
        assert schema is RepairDecision
        self.structured_method = method
        return self

    def invoke(self, messages: object, config: object = None) -> object:
        return self.result


def test_repair_subagent_uses_tool_calling_not_response_format() -> None:
    decision = RepairDecision(
        action=RepairAction.REWRITE,
        content="安全回答",
        reason_code=AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
    )
    model = FakeStructuredModel(decision)
    agent = KitchenAgent(
        model=model,  # type: ignore[arg-type]
        inventory=SimpleNamespace(),
        recipes=SimpleNamespace(),
        actions=SimpleNamespace(),
    )

    result = agent._run_repair_subagent(
        SupervisorRequest(content="已完成付款"),
        AuditResult(
            False,
            "blocked completion claim",
            AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        ),
    )

    assert result == decision
    assert model.structured_method == "json_mode"


def test_repair_prompt_contains_a_valid_rewrite_json_example() -> None:
    from app.prompts import REPAIR_SYSTEM_PROMPT

    assert (
        '{"action":"rewrite","content":"安全的重写回答",'
        '"feedback":null,"reason_code":"repairable_output_violation"}'
        in REPAIR_SYSTEM_PROMPT
    )


def test_tool_history_decodes_json_results_for_recipe_selection() -> None:
    from langchain_core.messages import ToolMessage

    history = KitchenAgent._tool_history([
        ToolMessage(name='search_recipes', tool_call_id='recipe-call',
                    content='[{"id":"local-1","title":"Tomato Egg"}]'),
        ToolMessage(name='propose_add_inventory_instance', tool_call_id='add-call',
                    content='{"action_id":"act_1","status":"pending"}'),
        ToolMessage(name='provider_message', tool_call_id='text-call',
                    content='plain text result'),
    ])
    assert history[0]['result'] == [{'id': 'local-1', 'title': 'Tomato Egg'}]
    assert history[1]['result'] == {'action_id': 'act_1', 'status': 'pending'}
    assert history[2]['result'] == 'plain text result'
