from io import BytesIO

import pytest
from PIL import Image

from app.context import ConfidenceStatus
from app.vision import classify_confidence, validate_image


def test_confidence_uses_two_thresholds() -> None:
    assert classify_confidence(0.90) == ConfidenceStatus.ACCEPTED
    assert classify_confidence(0.70) == ConfidenceStatus.NEEDS_CONFIRMATION
    assert classify_confidence(0.20) == ConfidenceStatus.REJECTED


def image_bytes(image_format: str, size: tuple[int, int] = (64, 64)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "green").save(buffer, format=image_format)
    return buffer.getvalue()


def test_ingredient_image_accepts_supported_format() -> None:
    validate_image(image_bytes("JPEG"))


def test_ingredient_image_rejects_webp_and_extreme_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="format"):
        validate_image(image_bytes("WEBP"))
    with pytest.raises(ValueError, match="aspect ratio"):
        validate_image(image_bytes("PNG", (64, 16)))
