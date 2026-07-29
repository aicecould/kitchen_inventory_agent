"""Persistent checkpoint for low-confidence image detections."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from app.context import AgentResult, ConfidenceStatus, Ingredient


class PendingVisionConfirmation(BaseModel):
    confirmation_id: str
    user_id: str
    request_text: str
    target_language: str
    detections: list[Ingredient]
    status: Literal["pending", "resolved", "expired"]
    created_at: datetime
    expires_at: datetime
    resolution: dict[str, object] | None = None
    result: AgentResult | None = None

    @property
    def low_confidence_candidates(self) -> list[Ingredient]:
        return [
            item for item in self.detections if item.status == ConfidenceStatus.REJECTED
        ]


class VisionConfirmationRepository:
    def __init__(self, db_path: Path, ttl_minutes: int = 15) -> None:
        self.db_path = db_path
        self.ttl_minutes = ttl_minutes

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_vision_confirmations (
                    confirmation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    resolution_json TEXT,
                    result_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_vision_confirmation_user_status "
                "ON pending_vision_confirmations(user_id, status)"
            )

    def create(
        self,
        *,
        user_id: str,
        request_text: str,
        target_language: str,
        detections: list[Ingredient],
    ) -> PendingVisionConfirmation:
        low = [item for item in detections if item.status == ConfidenceStatus.REJECTED]
        if not low:
            raise ValueError("A vision confirmation requires low-confidence candidates")
        now = datetime.now(timezone.utc)
        pending = PendingVisionConfirmation(
            confirmation_id=f"vision_{uuid4().hex}",
            user_id=user_id,
            request_text=request_text,
            target_language=target_language,
            detections=detections,
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=self.ttl_minutes),
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO pending_vision_confirmations(
                    confirmation_id, user_id, request_text, target_language,
                    detections_json, status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pending.confirmation_id,
                    pending.user_id,
                    pending.request_text,
                    pending.target_language,
                    json.dumps(
                        [item.model_dump(mode="json") for item in detections],
                        ensure_ascii=False,
                    ),
                    pending.status,
                    pending.created_at.isoformat(),
                    pending.expires_at.isoformat(),
                ),
            )
        return pending

    def get(self, confirmation_id: str, user_id: str) -> PendingVisionConfirmation:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM pending_vision_confirmations WHERE confirmation_id = ?",
                (confirmation_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Vision confirmation not found")
            pending = self._from_row(row)
            if pending.user_id != user_id:
                raise PermissionError("Vision confirmation belongs to another user")
            if (
                pending.status == "pending"
                and pending.expires_at <= datetime.now(timezone.utc)
            ):
                connection.execute(
                    "UPDATE pending_vision_confirmations SET status = 'expired' "
                    "WHERE confirmation_id = ?",
                    (confirmation_id,),
                )
                pending.status = "expired"
        return pending

    def complete(
        self,
        confirmation_id: str,
        user_id: str,
        *,
        resolution: dict[str, object],
        result: AgentResult,
    ) -> PendingVisionConfirmation:
        pending = self.get(confirmation_id, user_id)
        if pending.status != "pending":
            raise ValueError(
                f"Vision confirmation cannot be resolved from status: {pending.status}"
            )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE pending_vision_confirmations
                SET status = 'resolved', resolution_json = ?, result_json = ?
                WHERE confirmation_id = ? AND status = 'pending'
                """,
                (
                    json.dumps(resolution, ensure_ascii=False),
                    result.model_dump_json(),
                    confirmation_id,
                ),
            )
        completed = self.get(confirmation_id, user_id)
        if completed.status != "resolved":
            raise RuntimeError("Vision confirmation was resolved concurrently")
        return completed

    @staticmethod
    def _from_row(row: tuple[object, ...]) -> PendingVisionConfirmation:
        return PendingVisionConfirmation(
            confirmation_id=str(row[0]),
            user_id=str(row[1]),
            request_text=str(row[2]),
            target_language=str(row[3]),
            detections=[
                Ingredient.model_validate(item)
                for item in json.loads(str(row[4]))
            ],
            status=str(row[5]),  # type: ignore[arg-type]
            created_at=datetime.fromisoformat(str(row[6])),
            expires_at=datetime.fromisoformat(str(row[7])),
            resolution=json.loads(str(row[8])) if row[8] else None,
            result=AgentResult.model_validate_json(str(row[9])) if row[9] else None,
        )
