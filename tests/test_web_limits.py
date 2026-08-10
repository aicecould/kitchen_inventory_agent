from fastapi.testclient import TestClient

from app.limits import MAX_IMAGE_BYTES, MAX_TEXT_CHARS
from dataclasses import replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

import pytest

from app.config import get_settings
from app.web import app
from PIL import Image


client = TestClient(app)


class _ElementCollector(HTMLParser):
    """Collect the attributes of specifically identified HTTP-served elements."""

    def __init__(self) -> None:
        super().__init__()
        self.elements: dict[str, dict[str, str | None]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.elements[element_id] = attributes


def test_frontend_shell_serves_a_hidden_accessible_proactive_reminder() -> None:
    """The page shell exposes a quiet live region before its startup request resolves."""
    response = client.get("/")
    document = _ElementCollector()
    document.feed(response.text)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert document.elements["proactive-reminder"]["role"] == "status"
    assert "hidden" in document.elements["proactive-reminder"]


def test_frontend_loads_safe_markdown_renderer_before_the_application() -> None:
    """The browser must load the audited link renderer before app.js consumes it."""
    response = client.get("/")

    assert response.status_code == 200
    markdown_position = response.text.index('/static/markdown.js')
    app_position = response.text.index('/static/app.js')
    assert markdown_position < app_position


def test_inventory_api_serves_duplicate_instances_with_display_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The browser receives individual purchase instances, never an aggregated item."""
    import app.web as web
    from app.tools.inventory import InventoryRepository

    settings = replace(get_settings(), inventory_db_path=tmp_path / "inventory.db")
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    repository = InventoryRepository(settings.inventory_db_path)
    repository.initialize()
    stocked_at = datetime(2026, 8, 7, 9, 30, tzinfo=UTC)
    first = repository.add_instance("番茄", "2 个（熟）", None, "unknown", now=stocked_at)
    second = repository.add_instance(
        "番茄",
        "400g-500g",
        datetime(2026, 8, 12, 18, 0, tzinfo=UTC),
        "user",
        now=stocked_at,
    )

    response = TestClient(app, raise_server_exceptions=False).get("/api/inventory")

    assert response.status_code == 200
    assert response.json()["items"] == sorted(
        [first, second], key=lambda item: str(item["instance_id"])
    )
    assert set(response.json()["items"][0]) == {
        "instance_id",
        "name",
        "quantity_text",
        "stocked_at",
        "expires_at",
        "expiry_source",
        "last_confirmed_at",
    }


def test_actions_api_returns_structured_previews_for_all_inventory_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP confirmation boundary must serialize every operation's real preview."""
    import app.web as web
    from app.actions import InventoryActionService, PendingActionRepository
    from app.tools.inventory import InventoryRepository

    database = tmp_path / "inventory.db"
    inventory = InventoryRepository(database)
    inventory.initialize()
    repository = PendingActionRepository(database)
    repository.initialize()
    service = InventoryActionService(repository, inventory)
    first = inventory.add_instance("番茄", "2个", None, "unknown")
    second = inventory.add_instance("番茄", "半盒", None, "unknown")
    first_id = str(first["instance_id"])
    actions = [
        service.propose(
            web.SHOWCASE_USER_ID,
            "inventory.add",
            {
                "name": "牛奶",
                "quantity_text": "一盒",
                "expires_at": None,
                "expiry_source": "unknown",
            },
        ),
        service.propose(
            web.SHOWCASE_USER_ID,
            "inventory.replace",
            {
                "instance_id": first_id,
                "quantity_text": "1个",
                "expires_at": None,
                "expiry_source": "unknown",
            },
        ),
        service.propose(
            web.SHOWCASE_USER_ID,
            "inventory.remove",
            {"instance_id": first_id},
        ),
        service.propose(
            web.SHOWCASE_USER_ID,
            "inventory.remove_all",
            {"name": "番茄"},
        ),
        service.propose(
            web.SHOWCASE_USER_ID,
            "inventory.confirm",
            {"instance_id": first_id},
        ),
    ]
    monkeypatch.setattr(web, "get_action_service", lambda: service)

    response = client.get("/api/actions")

    assert response.status_code == 200
    payloads = {action["operation"]: action for action in response.json()["actions"]}
    assert set(payloads) == {
        "inventory.add",
        "inventory.replace",
        "inventory.remove",
        "inventory.remove_all",
        "inventory.confirm",
    }
    assert payloads["inventory.add"]["preview"] == actions[0].preview
    assert payloads["inventory.replace"]["preview"]["current"] == first
    assert payloads["inventory.remove"]["preview"] == {"current": first}
    assert payloads["inventory.remove_all"]["preview"] == {
        "name": "番茄",
        "items": [first, second],
        "expected_count": 2,
    }
    assert payloads["inventory.confirm"]["preview"] == {"current": first}


def test_cancel_action_returns_fixed_inventory_safe_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.web as web
    from app.actions import InventoryActionService, PendingActionRepository
    from app.tools.inventory import InventoryRepository

    database = tmp_path / "inventory.db"
    inventory = InventoryRepository(database)
    inventory.initialize()
    repository = PendingActionRepository(database)
    repository.initialize()
    service = InventoryActionService(repository, inventory)
    action = service.propose(
        web.SHOWCASE_USER_ID,
        "inventory.add",
        {
            "name": "牛奶",
            "quantity_text": "一盒",
            "expires_at": None,
            "expiry_source": "unknown",
        },
    )

    class RecordingPipeline:
        calls = 0

        def record_inventory_confirmation(self, **kwargs: object) -> None:
            self.calls += 1

    pipeline = RecordingPipeline()
    monkeypatch.setattr(web, "get_action_service", lambda: service)
    monkeypatch.setattr(web, "get_pipeline", lambda: pipeline)

    response = TestClient(app).post(f"/api/actions/{action.action_id}/cancel")

    assert response.status_code == 200
    assert response.json()["message"] == "库存操作已取消，库存未发生变化。"
    assert response.json()["action"]["status"] == "cancelled"
    assert inventory.find_by_name("牛奶") == []
    assert pipeline.calls == 1


def test_frontend_script_exposes_full_ids_and_all_confirmation_preview_labels() -> None:
    """The shipped browser code must render the authoritative preview contract."""
    script = client.get("/static/app.js").text

    assert "renderActionPreview(action)" in script
    assert "action.preview" in script
    assert "escapeHtml(item.instance_id)" in script
    assert "shortInstanceId(item.instance_id)" not in script
    assert "formatLocalDateTime(item.stocked_at)" in script
    assert "item.stocked_at || item.created_at" not in script
    assert 'data-preview-ready="${previewReady}"' in script
    assert 'previewReady ? "" : "disabled hidden aria-disabled=\\"true\\""' in script
    for label in (
        "库存新增",
        "库存替换",
        "单项删除",
        "同名全部删除",
        "库存校准",
        "预计删除数量",
        "日期来源",
    ):
        assert label in script


def test_status_exposes_request_limits() -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["limits"] == {
        "max_text_chars": MAX_TEXT_CHARS,
        "max_image_bytes": MAX_IMAGE_BYTES,
    }


def test_backend_rejects_oversized_text() -> None:
    response = client.post(
        "/api/process",
        data={"text": "x" * (MAX_TEXT_CHARS + 1), "language": "zh"},
    )

    assert response.status_code == 400
    assert "2000 个字符" in response.json()["detail"]


def test_backend_rejects_oversized_image_before_pipeline() -> None:
    response = client.post(
        "/api/process",
        data={"text": "识别图片", "language": "zh"},
        files={"image": ("large.jpg", b"x" * (MAX_IMAGE_BYTES + 1), "image/jpeg")},
    )

    assert response.status_code == 413
    assert "3 MiB" in response.json()["detail"]


def test_backend_rejects_unsupported_image_format_before_pipeline() -> None:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "green").save(buffer, format="WEBP")

    response = client.post(
        "/api/process",
        data={"text": "识别图片", "language": "zh"},
        files={"image": ("image.webp", buffer.getvalue(), "image/webp")},
    )

    assert response.status_code == 400
    assert "JPG、PNG 或 BMP" in response.json()["detail"]


def test_process_response_includes_backend_execution_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.web as web

    profile = tmp_path / "profile.md"
    profile.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile,
    )
    pipeline = web.build_pipeline(settings)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    monkeypatch.setattr(web, "_pipeline", pipeline)
    response = client.post(
        "/api/process",
        data={"text": "查询一下当前库存", "language": "zh"},
    )

    assert response.status_code == 200
    trace = response.json()["execution_trace"]
    assert trace[0]["name"] == "input_validation"
    assert trace[0]["status"] == "passed"
    assert trace[0]["duration_ms"] >= 0
    assert any(event["name"] == "list_inventory" for event in trace)
    assert any(event["name"] == "regex_audit" for event in trace)
    stored_trace = [
        event.model_dump(mode="json")
        for event in pipeline.history.list_records(web.SHOWCASE_USER_ID)[0].execution_trace
    ]
    assert stored_trace == trace


class FailureRecordingPipeline:
    def __init__(self) -> None:
        self.failures: list[dict[str, object]] = []

    def process_request(self, **kwargs: object) -> object:
        raise RuntimeError("provider secret must not leak")

    def resolve_vision_confirmation(self, **kwargs: object) -> object:
        raise RuntimeError("provider secret must not leak")

    def record_failure(self, **kwargs: object) -> None:
        self.failures.append(kwargs)


def test_process_provider_failure_is_recorded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.web as web

    pipeline = FailureRecordingPipeline()
    monkeypatch.setattr(web, "get_pipeline", lambda: pipeline)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/process",
        data={"text": "推荐番茄菜谱", "language": "zh"},
    )

    assert response.status_code == 502
    assert len(pipeline.failures) == 1
    assert pipeline.failures[0]["request_text"] == "推荐番茄菜谱"
    assert pipeline.failures[0]["response_text"] == response.json()["detail"]
    assert "secret" not in response.json()["detail"]


def test_oversized_text_failure_is_recorded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.web as web

    pipeline = FailureRecordingPipeline()
    monkeypatch.setattr(web, "get_pipeline", lambda: pipeline)
    text = "番" * (MAX_TEXT_CHARS + 1)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/process",
        data={"text": text, "language": "zh"},
    )

    assert response.status_code == 400
    assert len(pipeline.failures) == 1
    assert pipeline.failures[0]["request_text"] == text
    assert pipeline.failures[0]["response_text"] == response.json()["detail"]


def test_vision_resolution_failure_uses_confirmation_request_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.web as web

    pipeline = FailureRecordingPipeline()
    monkeypatch.setattr(web, "get_pipeline", lambda: pipeline)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/vision-confirmations/vision_123/resolve",
        json={"selections": []},
    )

    assert response.status_code == 502
    assert len(pipeline.failures) == 1
    assert pipeline.failures[0]["request_text"] == "图片确认：vision_123"
    assert pipeline.failures[0]["response_text"] == response.json()["detail"]


def test_missing_inventory_confirmation_is_recorded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.web as web

    class MissingActionService:
        def confirm(self, action_id: str, user_id: str) -> object:
            raise KeyError("Pending action not found")

    pipeline = FailureRecordingPipeline()
    monkeypatch.setattr(web, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(web, "get_action_service", lambda: MissingActionService())

    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/actions/action_missing/confirm"
    )

    assert response.status_code == 404
    assert len(pipeline.failures) == 1
    assert pipeline.failures[0]["request_text"] == (
        "确认库存操作：action_missing"
    )
    assert pipeline.failures[0]["response_text"] == response.json()["detail"]


def test_process_rejects_empty_bulk_delete_before_creating_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic no-match delete must be a safe not-found, not a proposal."""
    import app.web as web

    profile = tmp_path / "profile.md"
    profile.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile,
    )
    pipeline = web.build_pipeline(settings)
    monkeypatch.setattr(web, "_pipeline", pipeline)

    response = client.post(
        "/api/process",
        data={"text": "删除所有番茄", "language": "zh"},
    )

    assert response.status_code == 404
    assert "inventory instances not found: 番茄" in response.json()["detail"]
    assert pipeline.actions.repository.list_pending(web.SHOWCASE_USER_ID) == []


def test_allergen_api_persists_broad_and_custom_values(
    tmp_path: object, monkeypatch: object
) -> None:
    from pathlib import Path

    import app.web as web

    path = Path(str(tmp_path)) / "profile.md"
    path.write_text("# 用户画像\n\n## 饮食偏好\n\n- 少辣\n", encoding="utf-8")
    settings = replace(get_settings(), user_profile_path=path)
    monkeypatch.setattr(web, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    monkeypatch.setattr(web, "_pipeline", None)  # type: ignore[attr-defined]

    response = client.put(
        "/api/allergens",
        json={"broad": ["Dairy", "Peanut"], "custom": ["腰果", "虾皮"]},
    )
    loaded = client.get("/api/allergens")

    assert response.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["broad"] == ["Dairy", "Peanut"]
    assert loaded.json()["custom"] == ["腰果", "虾皮"]


def test_allergen_api_rejects_unsupported_category() -> None:
    response = client.put(
        "/api/allergens",
        json={"broad": ["NotARealCategory"], "custom": []},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (None, {"message": None}),
        ("鲜" * 99 + "。", {"message": "鲜" * 99 + "。"}),
    ],
)
def test_proactive_check_serializes_null_and_bounded_messages(
    monkeypatch: pytest.MonkeyPatch,
    message: str | None,
    expected: dict[str, str | None],
) -> None:
    import app.web as web

    class Service:
        def check(self, user_id: str) -> str | None:
            assert user_id == web.SHOWCASE_USER_ID
            return message

    monkeypatch.setattr(web, "get_proactive_service", lambda: Service())

    response = client.post("/api/proactive-check")

    assert response.status_code == 200
    assert response.json() == expected


def test_proactive_check_returns_null_when_service_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.web as web

    class FailingService:
        def check(self, user_id: str) -> str | None:
            raise RuntimeError("provider secret response")

    monkeypatch.setattr(web, "get_proactive_service", lambda: FailingService())

    response = client.post("/api/proactive-check")

    assert response.status_code == 200
    assert response.json() == {"message": None}
