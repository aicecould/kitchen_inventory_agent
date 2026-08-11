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
from app.memory import ConversationOutcome, read_user_profile
from app.pipeline import KitchenPipeline, build_pipeline, build_proactive_service
from app.proactive import ProactiveService
from app.actions import InventoryActionService, PendingActionRepository
from app.adapters.vision_api import VisionApiError
from app.tools.inventory import InventoryRepository
from app.vision import validate_image

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Kitchen Inventory Agent", version="0.1.0")
_pipeline: KitchenPipeline | None = None
_pipeline_lock = Lock()
_proactive_service: ProactiveService | None = None
_proactive_lock = Lock()
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


class VisionSelection(BaseModel):
    candidate_index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if "\n" in cleaned or "\r" in cleaned:
            raise ValueError("Ingredient name must be a single-line value")
        return cleaned


class VisionResolution(BaseModel):
    selections: list[VisionSelection] = Field(default_factory=list, max_length=20)


class TaskRecipeRequest(BaseModel):
    segment_id: str = Field(pattern=r"^hist_[0-9a-f]{32}$")
    recipe_id: str = Field(min_length=1, max_length=200)
    servings: int | None = Field(default=None, ge=1, le=100)
    expected_version: int | None = Field(default=None, ge=1)
    request_id: str = Field(min_length=1, max_length=100)


class TaskMutationRequest(BaseModel):
    expected_version: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=100)


class TaskServingsRequest(TaskMutationRequest):
    servings: int = Field(ge=1, le=100)


class ContextClearRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)


def get_pipeline() -> KitchenPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = build_pipeline()
    return _pipeline


def get_proactive_service() -> ProactiveService:
    global _proactive_service
    if _proactive_service is None:
        pipeline = get_pipeline()
        with _proactive_lock:
            if _proactive_service is None:
                _proactive_service = build_proactive_service(pipeline)
    return _proactive_service


def _record_failure(
    *,
    request_text: str,
    detail: str,
    trace: list[ExecutionTraceEvent] | None = None,
) -> None:
    get_pipeline().record_failure(
        user_id=SHOWCASE_USER_ID,
        request_text=request_text,
        response_text=detail,
        execution_trace=trace,
    )


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
        "food_classifier": (
            settings.food_classifier_enabled
            and settings.food_classifier_model_path.is_file()
            and settings.food_classifier_metadata_path.is_file()
        ),
        "local_recipe_search": (
            settings.recommender_enabled
            and settings.recommender_db_path.is_file()
            and settings.recommender_faiss_path.is_file()
            and settings.recommender_metadata_path.is_file()
            and settings.recommender_model_path.is_dir()
        ),
    }
    return {
        "ready": services["deepseek"],
        "services": services,
        "limits": {
            "max_text_chars": MAX_TEXT_CHARS,
            "max_image_bytes": MAX_IMAGE_BYTES,
        },
        "note": "订单暂未实现；回答监督采用一次子 Agent 修复和一次主 Agent 重试上限。",
    }


@app.get("/api/inventory")
def inventory_snapshot() -> dict[str, object]:
    settings = get_settings()
    repository = InventoryRepository(settings.inventory_db_path)
    repository.initialize()
    return {"items": repository.list_instances()}


@app.post("/api/proactive-check")
def proactive_check() -> dict[str, str | None]:
    try:
        message = get_proactive_service().check(SHOWCASE_USER_ID)
    except Exception:
        message = None
    return {"message": message}


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
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = build_pipeline(settings)
    pipeline = _pipeline
    assert pipeline is not None
    evidence = pipeline.history.append(
        user_id=SHOWCASE_USER_ID,
        request="确认更新过敏原设置",
        response="过敏原设置已保存。",
        confirmation_result={
            "type": "profile_allergens",
            "decision": "confirm",
            "broad": payload.broad,
            "custom": payload.custom,
        },
        outcome_type=ConversationOutcome.CONFIRMATION,
    )
    profile = pipeline.profile_actions.replace_confirmed_allergens(
        SHOWCASE_USER_ID,
        payload.broad,
        payload.custom,
        evidence_id=evidence.segment_id,
    )
    with _pipeline_lock:
        pipeline.profile = profile  # type: ignore[assignment]
    return {
        "broad": profile.allergen_intolerances,
        "custom": profile.custom_allergens,
    }


def _resolve_profile_proposal(proposal_id: str, decision: str) -> dict[str, object]:
    request_text = f"{'确认' if decision == 'confirm' else '拒绝'}画像变更：{proposal_id}"
    try:
        proposal = get_pipeline().resolve_profile_proposal(
            user_id=SHOWCASE_USER_ID,
            proposal_id=proposal_id,
            decision=decision,
        )
    except KeyError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=404, detail=detail) from exc
    except PermissionError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=403, detail=detail) from exc
    except ValueError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=409, detail=detail) from exc
    return {"proposal": proposal.model_dump(mode="json")}


@app.post("/api/profile-proposals/{proposal_id}/confirm")
def confirm_profile_proposal(proposal_id: str) -> dict[str, object]:
    return _resolve_profile_proposal(proposal_id, "confirm")


@app.post("/api/profile-proposals/{proposal_id}/reject")
def reject_profile_proposal(proposal_id: str) -> dict[str, object]:
    result = _resolve_profile_proposal(proposal_id, "reject")
    result["message"] = "画像变更已拒绝，长期画像未改变。"
    return result


def _task_error(exc: Exception) -> HTTPException:
    detail = str(exc)
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=detail)
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=detail)
    return HTTPException(status_code=409, detail=detail)


@app.get("/api/tasks")
def list_tasks() -> dict[str, object]:
    tasks = get_pipeline().tasks.repository.list_available(SHOWCASE_USER_ID)
    return {"tasks": [task.model_dump(mode="json") for task in tasks]}


@app.get("/api/tasks/active")
def active_task() -> dict[str, object]:
    task = get_pipeline().tasks.repository.get_active(SHOWCASE_USER_ID)
    return {"task": task.model_dump(mode="json") if task is not None else None}


def _confirm_task_recipe(
    payload: TaskRecipeRequest, *, as_new: bool
) -> dict[str, object]:
    pipeline = get_pipeline()
    try:
        task = pipeline.tasks.confirm_recipe(
            user_id=SHOWCASE_USER_ID,
            segment_id=payload.segment_id,
            recipe_id=payload.recipe_id,
            servings=payload.servings,
            default_context=pipeline.profile.default_context,
            as_new=as_new,
            expected_version=payload.expected_version,
            request_id=payload.request_id,
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise _task_error(exc) from exc
    return {"task": task.model_dump(mode="json")}


@app.post("/api/tasks/recipe")
def confirm_task_recipe(payload: TaskRecipeRequest) -> dict[str, object]:
    return _confirm_task_recipe(payload, as_new=False)


@app.post("/api/tasks/recipe-as-new")
def confirm_recipe_as_new(payload: TaskRecipeRequest) -> dict[str, object]:
    return _confirm_task_recipe(payload, as_new=True)


@app.patch("/api/tasks/{task_id}/servings")
def update_task_servings(
    task_id: str, payload: TaskServingsRequest
) -> dict[str, object]:
    try:
        task = get_pipeline().tasks.update_servings(
            user_id=SHOWCASE_USER_ID,
            task_id=task_id,
            servings=payload.servings,
            expected_version=payload.expected_version,
            request_id=payload.request_id,
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise _task_error(exc) from exc
    return {"task": task.model_dump(mode="json")}


@app.post("/api/tasks/{task_id}/activate")
def activate_task(
    task_id: str, payload: TaskMutationRequest
) -> dict[str, object]:
    try:
        task = get_pipeline().tasks.activate(
            user_id=SHOWCASE_USER_ID,
            task_id=task_id,
            expected_version=payload.expected_version,
            request_id=payload.request_id,
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise _task_error(exc) from exc
    return {"task": task.model_dump(mode="json")}


@app.post("/api/tasks/{task_id}/cooked")
def complete_task_as_cooked(
    task_id: str, payload: TaskMutationRequest
) -> dict[str, object]:
    try:
        task, changed = get_pipeline().tasks.complete_cooked(
            user_id=SHOWCASE_USER_ID,
            task_id=task_id,
            expected_version=payload.expected_version,
            request_id=payload.request_id,
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise _task_error(exc) from exc
    return {
        "task": task.model_dump(mode="json"),
        "start_reconciliation": changed,
    }


@app.post("/api/tasks/{task_id}/reconciliation/close")
def close_task_reconciliation(
    task_id: str, payload: TaskMutationRequest
) -> dict[str, object]:
    try:
        task = get_pipeline().tasks.close_reconciliation(
            user_id=SHOWCASE_USER_ID,
            task_id=task_id,
            expected_version=payload.expected_version,
            request_id=payload.request_id,
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise _task_error(exc) from exc
    return {"task": task.model_dump(mode="json")}


@app.post("/api/context/clear")
def clear_context(payload: ContextClearRequest) -> dict[str, object]:
    try:
        task = get_pipeline().tasks.clear_context(
            SHOWCASE_USER_ID, request_id=payload.request_id
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise _task_error(exc) from exc
    return {"task": task.model_dump(mode="json") if task is not None else None}


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
    request_text = f"确认库存操作：{action_id}"
    try:
        action = get_action_service().confirm(action_id, SHOWCASE_USER_ID)
    except KeyError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=404, detail=detail) from exc
    except PermissionError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=403, detail=detail) from exc
    except ValueError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=409, detail=detail) from exc
    get_pipeline().record_inventory_confirmation(action=action, decision="confirm")
    return {"action": action.model_dump(mode="json")}


@app.post("/api/actions/{action_id}/cancel")
def cancel_action(action_id: str) -> dict[str, object]:
    request_text = f"取消库存操作：{action_id}"
    try:
        action = get_action_service().cancel(action_id, SHOWCASE_USER_ID)
    except KeyError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=404, detail=detail) from exc
    except PermissionError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=403, detail=detail) from exc
    except ValueError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=409, detail=detail) from exc
    get_pipeline().record_inventory_confirmation(action=action, decision="cancel")
    return {
        "action": action.model_dump(mode="json"),
        "message": "库存操作已取消，库存未发生变化。",
    }


@app.post("/api/vision-confirmations/{confirmation_id}/resolve")
def resolve_vision_confirmation(
    confirmation_id: str,
    payload: VisionResolution,
) -> dict[str, object]:
    request_text = f"图片确认：{confirmation_id}"
    try:
        result = get_pipeline().resolve_vision_confirmation(
            user_id=SHOWCASE_USER_ID,
            confirmation_id=confirmation_id,
            selections=[
                selection.model_dump(mode="json")
                for selection in payload.selections
            ],
        )
    except KeyError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=404, detail=detail) from exc
    except PermissionError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=403, detail=detail) from exc
    except ValueError as exc:
        detail = str(exc)
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(status_code=409, detail=detail) from exc
    except Exception as exc:
        detail = (
            f"继续处理请求失败：{type(exc).__name__}。"
            "请检查本地 API 配置与服务状态。"
        )
        _record_failure(request_text=request_text, detail=detail)
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc
    return result.model_dump(mode="json")


@app.post("/api/process")
async def process_request(
    text: str = Form(..., min_length=1),
    language: str = Form("zh", min_length=2, max_length=10),
    image: UploadFile | None = File(default=None),
    reconciliation_task_id: str | None = Form(default=None),
) -> dict[str, object]:
    validation_started = perf_counter()
    if len(text) > MAX_TEXT_CHARS:
        detail = f"文字输入不能超过 {MAX_TEXT_CHARS} 个字符。"
        trace = [
            ExecutionTraceEvent(
                stage="request",
                name="input_validation",
                status="failed",
                detail="text too long",
                duration_ms=round((perf_counter() - validation_started) * 1000),
            )
        ]
        _record_failure(request_text=text, detail=detail, trace=trace)
        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            detail = f"图片不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MiB。"
            trace = [
                ExecutionTraceEvent(
                    stage="request",
                    name="input_validation",
                    status="failed",
                    detail="image too large",
                    duration_ms=round(
                        (perf_counter() - validation_started) * 1000
                    ),
                )
            ]
            _record_failure(request_text=text, detail=detail, trace=trace)
            raise HTTPException(
                status_code=413,
                detail=detail,
            )
        try:
            validate_image(image_bytes)
        except ValueError as exc:
            detail = str(exc)
            trace = [
                ExecutionTraceEvent(
                    stage="request",
                    name="input_validation",
                    status="failed",
                    detail="invalid image",
                    duration_ms=round(
                        (perf_counter() - validation_started) * 1000
                    ),
                )
            ]
            _record_failure(request_text=text, detail=detail, trace=trace)
            raise HTTPException(status_code=400, detail=detail) from exc
    validation_duration_ms = round((perf_counter() - validation_started) * 1000)
    validation_trace = ExecutionTraceEvent(
        stage="request",
        name="input_validation",
        status="passed",
        detail=(
            f"text={len(text)} character(s); "
            f"image={len(image_bytes) if image_bytes is not None else 0} byte(s)"
        ),
        duration_ms=validation_duration_ms,
    )
    try:
        result = get_pipeline().process_request(
            user_id=SHOWCASE_USER_ID,
            text=text,
            image_bytes=image_bytes,
            target_language=language,
            initial_trace=[validation_trace],
            reconciliation_task_id=reconciliation_task_id,
        )
    except VisionApiError as exc:
        detail = f"百度果蔬识别失败（错误码 {exc.code}）：{exc.user_message}"
        _record_failure(request_text=text, detail=detail, trace=[validation_trace])
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc
    except KeyError as exc:
        detail = str(exc)
        _record_failure(request_text=text, detail=detail, trace=[validation_trace])
        raise HTTPException(status_code=404, detail=detail) from exc
    except ValueError as exc:
        detail = str(exc)
        _record_failure(request_text=text, detail=detail, trace=[validation_trace])
        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:
        # Keep credentials and raw provider responses out of the browser.
        detail = (
            f"处理请求失败：{type(exc).__name__}。"
            "请检查本地 API 配置与服务状态。"
        )
        _record_failure(request_text=text, detail=detail, trace=[validation_trace])
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc
    return result.model_dump(mode="json")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
