from __future__ import annotations

import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from fastapi.testclient import TestClient

from app.web import app


ROOT_DIR = Path(__file__).resolve().parents[1]


def _run_node(expression: str) -> str:
    script = f"""
require('./frontend/markdown.js');
const tasks = require('./frontend/tasks.js');
{expression}
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def test_task_renderer_escapes_values_and_encodes_endpoint_ids() -> None:
    output = _run_node(
        """
const task = {
  task_id: 'task_/unsafe',
  status: 'active',
  query_text: '<script>alert(1)</script>',
  recipe: {title: 'Soup <b>hot</b>', ingredients: ['tomato']},
  servings: 2,
  version: 3,
  reconciliation_open: false,
};
console.log(tasks.renderTaskPanel([task]));
console.log(tasks.taskEndpoint(task.task_id, 'cooked'));
"""
    )

    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert "Soup &lt;b&gt;hot&lt;/b&gt;" in output
    assert "/api/tasks/task_%2Funsafe/cooked" in output


def test_recipe_request_builder_cannot_submit_recipe_content() -> None:
    output = _run_node(
        """
const payload = tasks.buildRecipeRequest({
  segmentId: 'hist_1',
  recipeId: 'recipe-1',
  servings: '2',
  expectedVersion: 4,
  requestId: 'request-1',
  title: 'forged title',
  ingredients: ['forged'],
  source_url: 'javascript:alert(1)',
});
console.log(JSON.stringify(payload));
"""
    )
    assert json.loads(output) == {
        "segment_id": "hist_1",
        "recipe_id": "recipe-1",
        "servings": 2,
        "expected_version": 4,
        "request_id": "request-1",
    }


def test_recipe_cards_and_reconciliation_fields_are_structured() -> None:
    output = _run_node(
        """
const html = tasks.renderRecipeOptions('hist_1', [{
  id: 'recipe-1',
  title: '<Tomato Soup>',
  ingredients: ['2 tomatoes'],
  source_url: 'https://recipes.example/soup',
}], {task_id: 'task_1', version: 2});
console.log(JSON.stringify({
  html,
  fields: tasks.reconciliationFields('task_1'),
}));
"""
    )
    result = json.loads(output)
    assert "&lt;Tomato Soup&gt;" in result["html"]
    assert 'data-recipe-id="recipe-1"' in result["html"]
    assert result["fields"] == {
        "text": "我已完成这道菜，请核对本次使用后的库存。",
        "language": "zh",
        "reconciliation_task_id": "task_1",
    }


class _PageElements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "script" and values.get("src"):
            self.scripts.append(str(values["src"]))


def test_served_page_exposes_minimal_task_attachment_before_app_script() -> None:
    response = TestClient(app).get("/")
    page = _PageElements()
    page.feed(response.text)

    assert response.status_code == 200
    assert {
        "task-panel",
        "task-list",
        "recipe-options",
        "clear-context",
    } <= page.ids
    assert page.scripts.index("/static/tasks.js") < page.scripts.index(
        "/static/app.js"
    )
