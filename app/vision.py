"""Image validation and vision API facade."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.adapters.vision_api import VisionApiClient
from app.context import ConfidenceStatus, Ingredient

DEFAULT_LOW_THRESHOLD = 0.50
DEFAULT_HIGH_THRESHOLD = 0.85


def validate_image(image_bytes: bytes, max_bytes: int = 8 * 1024 * 1024) -> None:
    if not image_bytes:
        raise ValueError("Image is empty")
    if len(image_bytes) > max_bytes:
        raise ValueError("Image exceeds size limit")
    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()


def classify_confidence(
    confidence: float,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> ConfidenceStatus:
    if confidence >= high_threshold:
        return ConfidenceStatus.ACCEPTED
    if confidence >= low_threshold:
        return ConfidenceStatus.NEEDS_CONFIRMATION
    return ConfidenceStatus.REJECTED


def recognize_ingredients(
    image_bytes: bytes,
    client: VisionApiClient,
) -> list[Ingredient]:
    validate_image(image_bytes)
    detections = client.recognize(image_bytes)
    return [
        Ingredient(
            name=item.name,
            confidence=item.confidence,
            status=classify_confidence(item.confidence),
        )
        for item in detections
    ]

