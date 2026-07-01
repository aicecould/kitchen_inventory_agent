"""Baidu general text translation API adapter."""

from __future__ import annotations

import hashlib
import secrets

import httpx


class TranslationApiClient:
    def __init__(
        self,
        endpoint: str,
        app_id: str,
        secret_key: str,
        timeout: float = 20,
    ) -> None:
        self.endpoint = endpoint
        self.app_id = app_id
        self.secret_key = secret_key
        self.timeout = timeout

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        if not text.strip() or source == target:
            return text
        salt = str(secrets.randbelow(900_000_000) + 100_000_000)
        raw_sign = f"{self.app_id}{text}{salt}{self.secret_key}"
        sign = hashlib.md5(raw_sign.encode("utf-8")).hexdigest()
        response = httpx.post(
            self.endpoint,
            data={
                "q": text,
                "from": source,
                "to": target,
                "appid": self.app_id,
                "salt": salt,
                "sign": sign,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "error_code" in payload:
            raise RuntimeError(f"Baidu translation failed: {payload}")
        return "\n".join(
            str(item.get("dst", "")) for item in payload.get("trans_result", [])
        ).strip()
