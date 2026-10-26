"""Centralized configuration loaded from the Git-ignored .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from recommender.model_config import EMBEDDING_MODEL_REVISION

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
    application_timezone: str
    http_timeout_seconds: float
    agent_recursion_limit: int
    deepseek_max_output_tokens: int
    deepseek_max_input_chars: int
    pending_action_ttl_minutes: int
    food_classifier_enabled: bool
    food_classifier_model_path: Path
    food_classifier_metadata_path: Path
    recommender_enabled: bool
    recommender_db_path: Path
    recommender_faiss_path: Path
    recommender_metadata_path: Path
    recommender_model_path: Path
    recommender_model_revision: str

    llm_log_source: str = "application"
    llm_logging_enabled: bool = True
    llm_log_db_path: Path = Path("data/llm_logs.db")
    llm_provider: str = "deepseek"
    qwen_base_url: str = "http://127.0.0.1:8001/v1"
    qwen_model: str = "Qwen/Qwen3-1.7B"
    qwen_api_key: str = "local"
    qwen_timeout_seconds: float = 120
    qwen_max_output_tokens: int = 800
    qwen_max_input_chars: int = 60000

    def __post_init__(self) -> None:
        if self.llm_provider not in {"deepseek", "qwen"}:
            raise ValueError("LLM_PROVIDER must be deepseek or qwen")

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "qwen":
            return bool(self.qwen_base_url.strip() and self.qwen_model.strip())
        return bool(self.deepseek_api_key.strip())

    @property
    def llm_model(self) -> str:
        return self.qwen_model if self.llm_provider == "qwen" else self.deepseek_model

    @property
    def llm_base_url(self) -> str:
        return self.qwen_base_url if self.llm_provider == "qwen" else self.deepseek_base_url

    @property
    def llm_max_input_chars(self) -> int:
        return self.qwen_max_input_chars if self.llm_provider == "qwen" else self.deepseek_max_input_chars

    def require(self, *names: str) -> None:
        missing = [name for name in names if not str(getattr(self, name, "")).strip()]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing configuration in .env: {joined}")


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek").strip().lower(),
        qwen_base_url=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8001/v1"),
        qwen_model=os.getenv("QWEN_MODEL", "Qwen/Qwen3-1.7B"),
        qwen_api_key=os.getenv("QWEN_API_KEY", "local"),
        qwen_timeout_seconds=float(os.getenv("QWEN_TIMEOUT_SECONDS", "120")),
        qwen_max_output_tokens=int(os.getenv("QWEN_MAX_OUTPUT_TOKENS", "800")),
        qwen_max_input_chars=int(os.getenv("QWEN_MAX_INPUT_CHARS", "60000")),
        llm_log_source=os.getenv("LLM_LOG_SOURCE", "application"),
        llm_logging_enabled=_env_bool("LLM_LOGGING_ENABLED", True),
        llm_log_db_path=Path(os.getenv("LLM_LOG_DB_PATH", "data/llm_logs.db")),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        baidu_image_api_key=os.getenv("BAIDU_IMAGE_API_KEY", ""),
        baidu_image_secret_key=os.getenv("BAIDU_IMAGE_SECRET_KEY", ""),
        baidu_image_endpoint=os.getenv(
            "BAIDU_IMAGE_ENDPOINT",
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient",
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
        application_timezone=os.getenv("APP_TIMEZONE", "Asia/Shanghai"),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "20")),
        agent_recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "10")),
        deepseek_max_output_tokens=int(
            os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "800")
        ),
        deepseek_max_input_chars=int(os.getenv("DEEPSEEK_MAX_INPUT_CHARS", "60000")),
        pending_action_ttl_minutes=int(
            os.getenv("PENDING_ACTION_TTL_MINUTES", "15")
        ),
        food_classifier_enabled=_env_bool("FOOD_CLASSIFIER_ENABLED", True),
        food_classifier_model_path=Path(
            os.getenv(
                "FOOD_CLASSIFIER_MODEL_PATH",
                "classifier/artifacts/food_query.bin",
            )
        ),
        food_classifier_metadata_path=Path(
            os.getenv(
                "FOOD_CLASSIFIER_METADATA_PATH",
                "classifier/artifacts/metadata.json",
            )
        ),
        recommender_enabled=_env_bool("RECOMMENDER_ENABLED", True),
        recommender_db_path=Path(
            os.getenv(
                "RECOMMENDER_DB_PATH",
                "recommender/artifacts/recipes.db",
            )
        ),
        recommender_faiss_path=Path(
            os.getenv(
                "RECOMMENDER_FAISS_PATH",
                "recommender/artifacts/recipes.faiss",
            )
        ),
        recommender_metadata_path=Path(
            os.getenv(
                "RECOMMENDER_METADATA_PATH",
                "recommender/artifacts/metadata.json",
            )
        ),
        recommender_model_path=Path(
            os.getenv("RECOMMENDER_MODEL_PATH", "recommender/model")
        ),
        recommender_model_revision=os.getenv(
            "RECOMMENDER_MODEL_REVISION",
            EMBEDDING_MODEL_REVISION,
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
