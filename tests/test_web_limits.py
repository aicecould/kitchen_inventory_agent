from fastapi.testclient import TestClient

from app.limits import MAX_IMAGE_BYTES, MAX_TEXT_CHARS
from app.web import app


client = TestClient(app)


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
    assert "8 MiB" in response.json()["detail"]
