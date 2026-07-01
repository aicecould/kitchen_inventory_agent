from app.context import ConfidenceStatus
from app.vision import classify_confidence


def test_confidence_uses_two_thresholds() -> None:
    assert classify_confidence(0.90) == ConfidenceStatus.ACCEPTED
    assert classify_confidence(0.70) == ConfidenceStatus.NEEDS_CONFIRMATION
    assert classify_confidence(0.20) == ConfidenceStatus.REJECTED

