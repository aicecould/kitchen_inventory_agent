"""Small FastAPI surface for the project showcase UI."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.allergens import OFFICIAL_INTOLERANCES, OFFICIAL_INTOLERANCE_SET
from app.config import get_settings
from app.context import ExecutionTraceEvent
from app.limits import MAX_IMAGE_BYTES, MAX_TEXT_CHARS
from app.memory import read_user_profile, write_user_allergens
from app.pipeline import KitchenPipeline, build_pipeline
from app.actions import InventoryActionService, PendingActionRepository
from app.tools.inventory import InventoryRepository

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Kitchen Inventory Agent", version="0.1.0")
_pipeline: KitchenPipeline | None = None
_pipeline_lock = Lock()
SHOWCASE_USER_ID = "showcase-user"


class AllergenSettings(BaseModel):
    broad: list[str] = Field(default_factory=list, max_length=12)
    custom: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("broad")
    @classmethod
    def validate_broad(cls, values: list[str]) -> list[str]:
        unique = list(dict.fromkeys(values))
        invalid = set(unique) - OFFICIAL_INTOLERANCE_SET
        if invalid:
            raise ValueError("Unsupported broad allergen category")
        return unique

    @field_validator("custom")
    @classmethod
    def validate_custom(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if any(len(value) > 100 for value in cleaned):
            raise ValueError("Custom allergen names must not exceed 100 characters")
        if any("\n" in value or "\r" in value for value in cleaned):
            raise ValueError("Custom allergen names must be single-line values")
        return cleaned


def get_pipeline() -> KitchenPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = build_pipeline()
    return _pipeline


@app.get("/api/status")
def service_status() -> dict[str, object]:
    settings = get_settings()
    services = {
        "deepseek": bool(settings.deepseek_api_key),
        "baidu_vision": bool(
            settings.baidu_image_api_key and settings.baidu_image_secret_key
        ),
        "spoonacular": bool(settings.spoonacular_api_key),
        "themealdb": bool(settings.themealdb_api_key),
    }
    return {
        "ready": services["deepseek"],
        "services": services,
        "limits": {
            "max_text_chars": MAX_TEXT_CHARS,
            "max_image_bytes": MAX_IMAGE_BYTES,
        },
        "note": "订单、向量意图识别与完整三级审核暂未实现。",
    }


@app.get("/api/inventory")
def inventory_snapshot() -> dict[str, object]:
    settings = get_settings()
    repository = InventoryRepository(settings.inventory_db_path)
    repository.initialize()
    return {"items": repository.list_items()}


@app.get("/api/allergens")
def allergen_settings() -> dict[str, object]:
    settings = get_settings()
    profile = read_user_profile(settings.user_profile_path)
    return {
        "options": list(OFFICIAL_INTOLERANCES),
        "broad": profile.allergen_intolerances,
        "custom": profile.custom_allergens,
    }


@app.put("/api/allergens")
def update_allergen_settings(payload: AllergenSettings) -> dict[str, object]:
    global _pipeline
    settings = get_settings()
    profile = write_user_allergens(
        settings.user_profile_path,
        payload.broad,
        payload.custom,
    )
    with _pipeline_lock:
        if _pipeline is not None:
            _pipeline.profile = profile
    return {
        "broad": profile.allergen_intolerances,
        "custom": profile.custom_allergens,
    }


def get_action_service() -> InventoryActionService:
    settings = get_settings()
    inventory = InventoryRepository(settings.inventory_db_path)
    inventory.initialize()
    action_repository = PendingActionRepository(settings.inventory_db_path)
    action_repository.initialize()
    return InventoryActionService(
        action_repository,
        inventory,
        ttl_minutes=settings.pending_action_ttl_minutes,
    )


@app.get("/api/actions")
def pending_actions() -> dict[str, object]:
    service = get_action_service()
    actions = service.repository.list_pending(SHOWCASE_USER_ID)
    return {"actions": [action.model_dump(mode="json") for action in actions]}


@app.post("/api/actions/{action_id}/confirm")
def confirm_action(action_id: str) -> dict[str, object]:
    try:
        action = get_action_service().confirm(action_id, SHOWCASE_USER_ID)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"action": action.model_dump(mode="json")}


@app.post("/api/actions/{action_id}/cancel")
def cancel_action(action_id: str) -> dict[str, object]:
    try:
        action = get_action_service().cancel(action_id, SHOWCASE_USER_ID)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"action": action.model_dump(mode="json")}


@app.post("/api/process")
async def process_request(
    text: str = Form(..., min_length=1),
    language: str = Form("zh", min_length=2, max_length=10),
    image: UploadFile | None = File(default=None),
) -> dict[str, object]:
    validation_started = perf_counter()
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"文字输入不能超过 {MAX_TEXT_CHARS} 个字符。",
        )

    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"图片不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MiB。",
            )
    validation_duration_ms = round((perf_counter() - validation_started) * 1000)
    try:
        result = get_pipeline().process_request(
            user_id=SHOWCASE_USER_ID,
            text=text,
            image_bytes=image_bytes,
            target_language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Keep credentials and raw provider responses out of the browser.
        raise HTTPException(
            status_code=502,
            detail=f"处理请求失败：{type(exc).__name__}。请检查本地 API 配置与服务状态。",
        ) from exc
    result.execution_trace.insert(
        0,
        ExecutionTraceEvent(
            stage="request",
            name="input_validation",
            status="passed",
            detail=(
                f"text={len(text)} character(s); "
                f"image={len(image_bytes) if image_bytes is not None else 0} byte(s)"
            ),
            duration_ms=validation_duration_ms,
        ),
    )
    return result.model_dump(mode="json")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
