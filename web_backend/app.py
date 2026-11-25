"""FastAPI application for the local browser/server edition of the platform."""

from __future__ import annotations

import sys
import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND_DIST = ROOT / "web_frontend" / "dist"

from web_backend.task_manager import TaskManager, TaskStateUnavailable
from web_backend.services.healing import routes as healing_routes
from web_backend.services.fdi import routes as fdi_routes
from web_backend.services.replay import routes as replay_routes
from web_backend.services import anomaly, cdq, classification, path_picker, scoring, sdg, sil
from platform_core.risk_control import FDI_TITLE, REPLAY_TITLE
from platform_core.risk_control.dnn_mpc.dataset import MAX_CSV_BYTES, example_csv
from platform_core.risk_control.service import ControlService
from platform_core.risk_control.presentation import (tokens as cstr_tokens, template as cstr_template,
                                                     trial_status as cstr_trial_status, online_update_status, control_status)


task_manager = TaskManager()
cstr_service = ControlService(task_manager)


class ScoreRequest(BaseModel):
    values: dict[str, float]
    weights: dict[str, float] = Field(default_factory=dict)


class SDGNodeRequest(BaseModel):
    id: str
    name: str
    type: Literal["R", "P", "C"]
    probability: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @field_validator("id", "name")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("节点 ID 和名称不能为空。")
        return value


class SDGEdgeRequest(BaseModel):
    source: str
    target: str
    type: Literal["+", "-"] = "+"
    probability: float = Field(..., ge=0, le=1, allow_inf_nan=False)

    @field_validator("source", "target")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("边的源节点和目标节点不能为空。")
        return value


class SDGRequest(BaseModel):
    nodes: list[SDGNodeRequest]
    edges: list[SDGEdgeRequest]


class CDQRequest(BaseModel):
    step: float = 1.0
    horizon: int = 10
    sample_index: int = 0
    cv: list[float] | None = None
    u_now: list[float] | None = None
    u_after: list[float] | None = None

    @field_validator("cv")
    @classmethod
    def validate_cv_bounds(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        for index, item in enumerate(value):
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ValueError(f"CV特征向量第{index + 1}项必须是有限数值")
            if not cdq.CDQ_VECTOR_MIN <= numeric <= cdq.CDQ_VECTOR_MAX:
                raise ValueError(
                    f"CV特征向量第{index + 1}项当前值为{numeric:g}，超出参数边界。"
                )
        return value


class SILTaskRequest(BaseModel):
    m: int = Field(default=2, ge=1, le=10)
    n: int = Field(default=4, ge=1, le=10)
    lambda_fit: float = Field(default=111.11, ge=0.01, le=100000, allow_inf_nan=False)
    ti: float = Field(default=8760, ge=1, le=100000, allow_inf_nan=False)
    mrt: float = Field(default=8, ge=0, le=10000, allow_inf_nan=False)
    nsim: int = Field(default=500, ge=1, le=2000)
    years: int = Field(default=10000, ge=1001, le=100000)
    ccf_mode: Literal["total", "partial"] = "total"
    total_beta: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    partial_betas: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_architecture_and_probabilities(self) -> "SILTaskRequest":
        if self.m > self.n:
            raise ValueError("表决架构必须满足 M ≤ N。")
        for order, beta in self.partial_betas.items():
            if not math.isfinite(float(beta)):
                raise ValueError(f"部分 β{order} 必须是有限数值。")
            if not 0 <= float(beta) <= 1:
                raise ValueError(f"部分 β{order} 必须在 0 到 1 之间（含边界）。")
        if self.ccf_mode == "partial" and sum(float(value) for value in self.partial_betas.values()) >= 1:
            raise ValueError("部分 β 的总和必须小于 1。")
        return self


class ClassificationTaskRequest(BaseModel):
    dataset: Literal["original", "easy", "hard"] = "original"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001


class AnomalyTaskRequest(BaseModel):
    mcr_root: str = r"E:\MATLAB2024"
    attack_min_pct: float = Field(default=5.0, ge=5.0, le=50.0, allow_inf_nan=False)
    attack_max_pct: float = Field(default=10.0, ge=5.0, le=50.0, allow_inf_nan=False)
    measurement_noise_pct: float = Field(default=2.0, ge=1.0, le=30.0, allow_inf_nan=False)
    process_disturbance_pct: float = Field(default=5.0, ge=1.0, le=30.0, allow_inf_nan=False)

    @field_validator("mcr_root")
    @classmethod
    def validate_mcr_root(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("请填写 MATLAB Runtime 根目录。")
        return normalized

    @model_validator(mode="after")
    def validate_attack_range(self) -> "AnomalyTaskRequest":
        if self.attack_min_pct > self.attack_max_pct:
            raise ValueError("攻击幅度最小值不能大于最大值。")
        return self


class LocalPathSelectRequest(BaseModel):
    kind: Literal["directory", "open-mat", "save-mat", "open-npz", "save-npz"]
    initial_path: str = ""


class CstrCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = "default"
    auto_start: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)


class CstrCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["start", "pause", "resume", "reset", "end", "reference", "model", "disturbance", "clear_disturbance"]
    key: str = Field(min_length=1, max_length=128)
    preset: str | None = None
    duration_cycles: int | None = Field(default=None, strict=True)
    value: float | None = None
    model_id: str | None = None
    settings: dict[str, Any] | None = None


class CstrUpdate(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"


class CstrUpdatePolicy(CstrUpdate):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(strict=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global task_manager, cstr_service
    if task_manager._closed:
        task_manager = TaskManager()
    cstr_service = ControlService(task_manager)
    yield
    cstr_service.close()
    task_manager.shutdown()


app = FastAPI(title="流程行业动态风险管控工具集平台（本机Web版）", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@app.exception_handler(TaskStateUnavailable)
async def task_state_exception_handler(_, exc: TaskStateUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "mode": "local-only", "host": "127.0.0.1", "frontend_built": (FRONTEND_DIST / "index.html").is_file()}


@app.post("/api/local-paths/select")
def select_local_path(request: LocalPathSelectRequest) -> dict[str, str]:
    try:
        selected = path_picker.choose_local_path(request.kind, request.initial_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"path": selected}



@app.get("/api/modules")
def modules() -> list[dict[str, str]]:
    return [
        {"id": "governance", "name": "异构数据治理", "status": "placeholder"},
        {"id": "score", "name": "多评估准则融合的风险学习分析", "status": "available"},
        {"id": "classification", "name": "潜在安全威胁识别与自动分类", "status": "available"},
        {"id": "cdq", "name": "风险场景动态匹配与适配方案生成算法", "status": "available"},
        {"id": "sdg", "name": "SIS自主化检测 · SDG-HAZOP", "status": "available"},
        {"id": "sil", "name": "在线SIL验证", "status": "available"},
        {"id": "anomaly", "name": "异常行为检测", "status": "available"},
        {"id": "risk-overview", "name": "综合仿真展示", "status": "placeholder"},
        {"id": "dnn-mpc", "name": "深度神经网络优化控制", "status": "available"},
        {"id": "self-healing", "name": "多变量过程自愈控制", "status": "available"},
        {"id": "fdi", "name": FDI_TITLE, "status": "available"},
        {"id": "replay", "name": REPLAY_TITLE, "status": "available"},
    ]


@app.get("/api/score/config")
def score_config() -> dict[str, Any]:
    return {"metrics": scoring.METRIC_CONFIGS, "defaults": scoring.default_values(), "default_weights": {name: 1.0 for name in scoring.METRIC_CONFIGS}}


@app.post("/api/score/random")
def score_random() -> dict[str, float]:
    return scoring.random_values()


@app.post("/api/score/evaluate")
def score_evaluate(request: ScoreRequest) -> dict[str, Any]:
    try:
        return scoring.evaluate(request.values, request.weights)
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.get("/api/sdg/example")
def sdg_example() -> dict[str, Any]:
    return {"nodes": sdg.DEFAULT_NODES, "edges": sdg.DEFAULT_EDGES}


@app.get("/api/sdg/config")
def sdg_config() -> dict[str, Any]:
    try:
        return sdg.config()
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/sdg/analyze")
def sdg_analyze(request: SDGRequest) -> dict[str, Any]:
    try:
        return sdg.analyze([node.model_dump() for node in request.nodes], [edge.model_dump() for edge in request.edges])
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.get("/api/cdq/config")
def cdq_config() -> dict[str, Any]:
    try:
        return cdq.dataset_summary()
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/cdq/analyze")
def cdq_analyze(request: CDQRequest) -> dict[str, Any]:
    try:
        cdq.validate_request_vectors(request.cv, request.u_now, request.u_after)
        return cdq.analyze(request.step, request.horizon, request.sample_index, request.cv, request.u_now, request.u_after)
    except cdq.CDQValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.get("/api/sil/defaults")
def sil_defaults() -> dict[str, Any]:
    return {"m": 2, "n": 4, "lambda_fit": 111.11, "ti": 8760, "mrt": 8, "nsim": 500, "years": 10000, "ccf_mode": "total", "total_beta": 0.1, "partial_betas": {"2": 0.0333, "3": 0.0333, "4": 0.0333}}


@app.post("/api/sil/tasks")
def create_sil_task(request: SILTaskRequest) -> dict[str, str]:
    config = request.model_dump()
    task = task_manager.submit("GSPN-MC SIL验证", lambda log: sil.validate(config, log))
    return {"task_id": task.id, "status": task.status}


@app.post("/api/anomaly/tasks")
def create_anomaly_task(request: AnomalyTaskRequest) -> dict[str, str]:
    config = request.model_dump()
    task = task_manager.submit("基于移动目标防御的异常检测", lambda log: anomaly.run(config, log))
    return {"task_id": task.id, "status": task.status}


@app.get("/api/anomaly/images/{image_name}")
def anomaly_image(image_name: str) -> FileResponse:
    allowed = {"topology.png", "detection_probability.png"}
    if image_name not in allowed:
        raise HTTPException(status_code=404, detail="异常检测图片不存在")
    image_path = anomaly.FIGURE_DIR / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="异常检测尚未生成该图片")
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/classification/datasets")
def classification_datasets() -> list[dict[str, str]]:
    return classification.dataset_options()


@app.post("/api/classification/tasks")
def create_classification_task(request: ClassificationTaskRequest) -> dict[str, str]:
    config = request.model_dump()
    task = task_manager.submit("潜在安全威胁分类训练", lambda log: classification.train(config, log))
    return {"task_id": task.id, "status": task.status}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或任务记录已被移除")
    return task


@app.post("/api/tasks/{task_id}/cancel")
def cancel_control_task(task_id: str):
    try:
        current = task_manager.get(task_id)
        if current and current["title"] == "CSTR 在线模型更新":
            raise HTTPException(403, "请通过所属控制会话取消更新")
        task = task_manager.cancel(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或本机服务已重启")
    return task


def _cstr_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise _bad_request(exc) from exc


@app.get("/api/cstr/config")
def cstr_config():
    return {**cstr_service.config(), "diagram_template": cstr_template()}


@app.post("/api/cstr/training")
def cstr_train(options: dict[str, Any]):
    return {"task_id": _cstr_call(cstr_service.train, options)}


@app.get("/api/cstr/training/example.csv")
def cstr_example():
    return Response(example_csv(), media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="cstr-trajectories.csv"'})


@app.post("/api/cstr/training/csv")
async def cstr_train_csv(request: Request, epochs: int = 320, seed: int = 42,
                         lr: float = 0.001, device: str = "auto"):
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_CSV_BYTES:
            raise HTTPException(413, "CSV 文件超过 32 MiB")
        content.extend(chunk)
    if not content:
        raise HTTPException(400, "请选择非空 CSV 文件")
    return {"task_id": _cstr_call(cstr_service.train,
            dict(epochs=epochs, seed=seed, lr=lr, device=device), bytes(content))}


@app.get("/api/cstr/training/{task_id}/report")
def cstr_training_report(task_id: str):
    task = get_task(task_id)
    directory = task_manager.artifact_dir(task_id)
    if task["title"] != "CSTR 预测模型训练" or not directory or not (directory / "result.json").is_file():
        raise HTTPException(404, "训练报告尚不可用")
    return FileResponse(directory / "result.json", filename="cstr-training.json")


@app.post("/api/cstr/sessions")
def cstr_create(request: CstrCreate):
    s = _cstr_call(cstr_service.create, model_id=request.model_id, settings=request.settings, lease=True, auto_start=request.auto_start)
    return {"id": s.id, "owner": s.owner, "run": s.run}


@app.get("/api/cstr/sessions/{identifier}")
def cstr_snapshot(identifier: str, after: int = -1, owner: str | None = Header(None, alias="X-Control-Owner")):
    s = _cstr_call(cstr_service.session, identifier)
    data = s.snapshot(after, owner)
    online = data["online_update"]
    return {**data, "control_display": control_status(data), "display": cstr_tokens(data["latest"]), "disturbance_display": cstr_trial_status(data["disturbance"]),
            "online_update": {**online, **online_update_status(online)}}


@app.get("/api/cstr/sessions/{identifier}/presence")
async def cstr_presence(identifier: str, request: Request, owner: str | None = Header(None, alias="X-Control-Owner")):
    session = _cstr_call(cstr_service.session, identifier, owner, True)
    return _presence_stream(session, request)


def _presence_stream(session, request):

    async def connected():
        session.owner_connected()
        try:
            while not session.closed and not await request.is_disconnected():
                yield b": online\n\n"
                await asyncio.sleep(5)
        finally:
            session.owner_disconnected()

    return StreamingResponse(connected(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


app.include_router(healing_routes(lambda: cstr_service.healing, _cstr_call, _presence_stream))
app.include_router(fdi_routes(lambda: cstr_service.fdi, _cstr_call, _presence_stream))
app.include_router(replay_routes(lambda: cstr_service.replay, _cstr_call, _presence_stream))


@app.get("/api/cstr/sessions/{identifier}/history")
def cstr_history(identifier: str, run: str, first: int, count: int):
    session = _cstr_call(cstr_service.session, identifier)
    return {"run": run, "samples": _cstr_call(session.sample_window, run, first, count)}


@app.post("/api/cstr/sessions/{identifier}/commands")
def cstr_command(identifier: str, request: CstrCommand, owner: str | None = Header(None, alias="X-Control-Owner")):
    s = _cstr_call(cstr_service.session, identifier, owner, True)
    return _cstr_call(cstr_service.command, s, request.kind, request.key, request.value,
                      request.model_id, request.settings, preset=request.preset, duration_cycles=request.duration_cycles)


@app.post("/api/cstr/sessions/{identifier}/updates")
def cstr_update(identifier: str, request: CstrUpdate, owner: str | None = Header(None, alias="X-Control-Owner")):
    s = _cstr_call(cstr_service.session, identifier, owner, True)
    return {"task_id": _cstr_call(cstr_service.update_model, s, request.device)}


@app.post("/api/cstr/sessions/{identifier}/updates/policy")
def cstr_update_policy(identifier: str, request: CstrUpdatePolicy, owner: str | None = Header(None, alias="X-Control-Owner")):
    session = _cstr_call(cstr_service.session, identifier, owner, True)
    policy = _cstr_call(cstr_service.configure_updates, session, request.enabled, request.device)
    return {**policy, **online_update_status(policy)}


@app.post("/api/cstr/sessions/{identifier}/updates/{task_id}/cancel")
def cstr_cancel_update(identifier: str, task_id: str, owner: str | None = Header(None, alias="X-Control-Owner")):
    s = _cstr_call(cstr_service.session, identifier, owner, True)
    with s.lock:
        if not s.update or s.update["id"] != task_id:
            raise HTTPException(409, "此更新已结束或不属于本次运行")
    return _cstr_call(cstr_service.cancel, task_id)


@app.get("/api/cstr/sessions/{identifier}/export")
def cstr_export(identifier: str):
    s = _cstr_call(cstr_service.session, identifier)
    return JSONResponse(s.export(), headers={"Content-Disposition": 'attachment; filename="cstr-run.json"'})


@app.delete("/api/cstr/sessions/{identifier}")
def cstr_release(identifier: str, owner: str | None = Header(None, alias="X-Control-Owner")):
    s = _cstr_call(cstr_service.session, identifier, owner, True)
    cstr_service.release(s)
    return {"status": "closed"}


@app.get("/{requested_path:path}", include_in_schema=False)
def frontend(requested_path: str):
    if requested_path.startswith("api/"):
        raise HTTPException(404, "接口不存在")
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            status_code=503,
            content={"detail": "Web前端尚未构建。请在web_frontend目录运行 npm install 和 npm run build，或执行 scripts/run_web_local.ps1。"},
        )
    candidate = (FRONTEND_DIST / requested_path).resolve()
    if requested_path and candidate.is_file() and str(candidate).startswith(str(FRONTEND_DIST.resolve())):
        return FileResponse(candidate)
    # The shell is the version selector for hashed Vite assets. Prevent a
    # previously opened tab from keeping the old CS/BS layout indefinitely.
    return FileResponse(index_path, headers={"Cache-Control": "no-store, max-age=0"})
