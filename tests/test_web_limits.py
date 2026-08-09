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
    assert response.json()["items"] == [first, second]


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


def test_process_response_includes_backend_execution_trace() -> None:
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
