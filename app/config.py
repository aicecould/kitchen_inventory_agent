"""Centralized configuration loaded from the Git-ignored .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    baidu_image_api_key: str
    baidu_image_secret_key: str
    baidu_image_endpoint: str
    spoonacular_api_key: str
    spoonacular_base_url: str
    themealdb_api_key: str
    themealdb_base_url: str
    inventory_db_path: Path
    user_profile_path: Path
    http_timeout_seconds: float
    agent_recursion_limit: int
    deepseek_max_output_tokens: int
    deepseek_max_input_chars: int
    pending_action_ttl_minutes: int

    def require(self, *names: str) -> None:
        missing = [name for name in names if not str(getattr(self, name, "")).strip()]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing configuration in .env: {joined}")


def get_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        baidu_image_api_key=os.getenv("BAIDU_IMAGE_API_KEY", ""),
        baidu_image_secret_key=os.getenv("BAIDU_IMAGE_SECRET_KEY", ""),
        baidu_image_endpoint=os.getenv(
            "BAIDU_IMAGE_ENDPOINT",
            "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general",
        ),
        spoonacular_api_key=os.getenv("SPOONACULAR_API_KEY", ""),
        spoonacular_base_url=os.getenv(
            "SPOONACULAR_BASE_URL", "https://api.spoonacular.com"
        ),
        themealdb_api_key=os.getenv("THEMEALDB_API_KEY", "1"),
        themealdb_base_url=os.getenv(
            "THEMEALDB_BASE_URL", "https://www.themealdb.com/api/json/v1"
        ),
        inventory_db_path=Path(os.getenv("INVENTORY_DB_PATH", "data/inventory.db")),
        user_profile_path=Path(
            os.getenv("USER_PROFILE_PATH", "data/user_profile.md")
        ),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "20")),
        agent_recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "10")),
        deepseek_max_output_tokens=int(
            os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "800")
        ),
        deepseek_max_input_chars=int(os.getenv("DEEPSEEK_MAX_INPUT_CHARS", "12000")),
        pending_action_ttl_minutes=int(
            os.getenv("PENDING_ACTION_TTL_MINUTES", "15")
        ),
    )
