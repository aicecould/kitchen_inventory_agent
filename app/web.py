"""Small FastAPI surface for the project showcase UI."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.pipeline import KitchenPipeline, build_pipeline
from app.actions import InventoryActionService, PendingActionRepository
from app.tools.inventory import InventoryRepository

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Kitchen Inventory Agent", version="0.1.0")
_pipeline: KitchenPipeline | None = None
_pipeline_lock = Lock()
SHOWCASE_USER_ID = "showcase-user"


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
        "baidu_translate": bool(
            settings.baidu_translate_app_id and settings.baidu_translate_secret_key
        ),
        "spoonacular": bool(settings.spoonacular_api_key),
        "themealdb": bool(settings.themealdb_api_key),
    }
    return {
        "ready": services["deepseek"],
        "services": services,
        "note": "订单、向量意图识别与完整三级审核暂未实现。",
    }


@app.get("/api/inventory")
def inventory_snapshot() -> dict[str, object]:
    settings = get_settings()
    repository = InventoryRepository(settings.inventory_db_path)
    repository.initialize()
    return {"items": repository.list_items()}


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
    text: str = Form(..., min_length=1, max_length=2_000),
    language: str = Form("zh", min_length=2, max_length=10),
    image: UploadFile | None = File(default=None),
) -> dict[str, object]:
    image_bytes = await image.read() if image is not None else None
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
    return result.model_dump(mode="json")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
