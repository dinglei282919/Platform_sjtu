"""FastAPI application for the local browser/server edition of the platform."""

from __future__ import annotations

import sys
import math
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND_DIST = ROOT / "web_frontend" / "dist"

from web_backend.task_manager import TaskManager, TaskStateUnavailable
from web_backend.services import anomaly, cdq, classification, fdi, replay, self_healing, mpc, path_picker, scoring, sdg, sil, training


task_manager = TaskManager()


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


class TrainingTaskRequest(BaseModel):
    model_config = {"extra": "forbid"}
    output_dir: str = Field(default_factory=lambda: training.default_config()["output_dir"])
    model_path: str = Field(default_factory=lambda: training.default_config()["model_path"])
    dataset_path: str = Field(default_factory=lambda: training.default_config()["dataset_path"])
    sample_count: int = Field(default=1000, ge=100, le=100000, strict=True)
    epochs: int = Field(default=200, ge=1, le=5000, strict=True)
    lr: float = Field(default=0.001, ge=0.000001, le=0.1, allow_inf_nan=False)
    hidden_layers: str = "64,64"

    @model_validator(mode="after")
    def validate_python_config(self):
        from platform_core.risk_control.dnn_mpc.config import validate_config
        normalized = validate_config("training", self.model_dump())
        for key, value in normalized.items():
            setattr(self, key, value)
        return self


class MpcTaskRequest(BaseModel):
    model_config = {"extra": "forbid"}
    output_dir: str = Field(default_factory=lambda: mpc.default_config()["output_dir"])
    model_path: str = Field(default_factory=lambda: mpc.default_config()["model_path"])
    sim_time: float = Field(default=5.0, ge=0.2, le=20.0, allow_inf_nan=False)
    prediction_horizon: int = Field(default=10, ge=1, le=60, strict=True)

    @model_validator(mode="after")
    def validate_python_config(self):
        from platform_core.risk_control.dnn_mpc.config import validate_config
        validate_config("mpc", self.model_dump())
        return self

class FdiTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sim_time: float = Field(default=10., ge=.1, le=10., allow_inf_nan=False, strict=True)

class ReplayTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sim_time: float = Field(default=50., ge=.1, le=50., allow_inf_nan=False, strict=True)


class SelfHealingTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("config")
    @classmethod
    def validate_experiment(cls, value):
        return self_healing.validate_config(value)



class LocalPathSelectRequest(BaseModel):
    kind: Literal["directory", "open-mat", "save-mat", "open-npz", "save-npz"]
    initial_path: str = ""


class DnnModelInfoRequest(BaseModel):
    model_path: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
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
        {"id": "dnn-mpc", "name": "DNN/MPC 控制训练与优化", "status": "available"},
        {"id": "fdi", "name": "虚假数据注入下的自适应控制仿真", "status": "available"},
        {"id": "replay", "name": "重放攻击下的自适应控制仿真", "status": "available"},
        {"id": "self-healing", "name": "多变量过程自愈控制仿真", "status": "available"},
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


@app.get("/api/training/defaults")
def training_defaults() -> dict[str, Any]:
    return training.default_config()


@app.post("/api/training/tasks")
def create_training_task(request: TrainingTaskRequest) -> dict[str, str]:
    config = request.model_dump()
    from platform_core.risk_control.dnn_mpc.runner import new_run_dir
    output_dir = new_run_dir(config)
    task = task_manager.submit(
        "风险场景下控制系统智能模型训练算法 · DNNTrain",
        lambda log: training.run(config, log, run_dir=output_dir),
        progress_path=output_dir / "progress.json",
        artifact_dir=output_dir,
    )
    return {"task_id": task.id, "status": task.status}


@app.get("/api/training/images/{image_name}")
def training_image(image_name: str) -> FileResponse:
    try:
        image = training.image_path(image_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image, media_type="image/svg+xml")


@app.get("/api/training/tasks/{task_id}/images/{image_name}")
def training_task_image(task_id: str, image_name: str) -> FileResponse:
    output_dir = task_manager.artifact_dir(task_id)
    if output_dir is None:
        raise HTTPException(status_code=404, detail="训练任务不存在或本机服务已重启")
    try:
        image = training.image_path(image_name, output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image, media_type="image/svg+xml")


@app.get("/api/mpc/defaults")
def mpc_defaults() -> dict[str, Any]:
    return mpc.default_config()


@app.post("/api/mpc/model-info")
def mpc_model_info(request: DnnModelInfoRequest) -> dict[str, Any]:
    from platform_core.risk_control.dnn_mpc.config import model_info
    try:
        return model_info(request.model_path)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/mpc/tasks")
def create_mpc_task(request: MpcTaskRequest) -> dict[str, str]:
    config = request.model_dump()
    from platform_core.risk_control.dnn_mpc.runner import new_run_dir
    output_dir = new_run_dir(config)
    task = task_manager.submit(
        "风险场景下智能模型优化控制算法 · MPC simulation",
        lambda log: mpc.run(config, log, run_dir=output_dir),
        progress_path=output_dir / "progress.json",
        artifact_dir=output_dir,
    )
    return {"task_id": task.id, "status": task.status}


@app.get("/api/mpc/images/{image_name}")
def mpc_image(image_name: str) -> FileResponse:
    try:
        image = mpc.image_path(image_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image, media_type="image/svg+xml")


@app.get("/api/mpc/tasks/{task_id}/images/{image_name}")
def mpc_task_image(task_id: str, image_name: str) -> FileResponse:
    output_dir = task_manager.artifact_dir(task_id)
    if output_dir is None:
        raise HTTPException(status_code=404, detail="MPC 任务不存在或本机服务已重启")
    try:
        image = mpc.image_path(image_name, output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image, media_type="image/svg+xml")


@app.get("/api/dnn/{mode}/tasks/{task_id}/files/{file_name}")
def dnn_artifact(mode: Literal["training", "mpc"], task_id: str, file_name: str) -> FileResponse:
    from platform_core.risk_control.dnn_mpc.runner import artifact_path
    directory = task_manager.artifact_dir(task_id)
    if directory is None:
        raise HTTPException(status_code=404, detail="任务不存在或任务记录已被移除")
    try:
        path = artifact_path(mode, file_name, directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = {".svg": "image/svg+xml", ".pdf": "application/pdf", ".csv": "text/csv",
                  ".json": "application/json", ".zip": "application/zip"}.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


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
        task = task_manager.cancel(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或本机服务已重启")
    return task


@app.get("/api/fdi/defaults")
def fdi_defaults():
    return fdi.default_config()


@app.post("/api/fdi/tasks")
def create_fdi_task(request: FdiTaskRequest):
    directory = fdi.new_run_dir()
    config = request.model_dump()
    cancelled = Event()
    task = task_manager.submit("虚假数据注入下的自适应控制仿真 · FDI",
                               lambda log: fdi.run(config, log, run_dir=directory, cancelled=cancelled.is_set),
                               artifact_dir=directory, progress_path=directory / "progress.json", cancel_event=cancelled)
    return {"task_id": task.id, "status": task.status}


def _fdi_artifact_path(task_id, name, *, preview=False):
    if not fdi.is_artifact(name):
        raise HTTPException(status_code=404, detail="FDI 结果文件不存在。")
    task = task_manager.get(task_id)
    directory = task_manager.artifact_dir(task_id)
    if task is None or directory is None:
        raise HTTPException(status_code=404, detail="FDI 任务不存在或本机服务已重启，请重新运行。")
    if task["status"] != "succeeded":
        progress = task.get("progress") or {}
        if not (preview and name.endswith(".svg") and progress.get("module") == "fdi"
                and progress.get("outputs", {}).get(Path(name).stem)):
            raise HTTPException(status_code=409, detail="仿真尚未成功完成。")
    elif (task.get("result") or {}).get("module") != "fdi":
        raise HTTPException(status_code=404, detail="该任务没有 FDI 结果。")
    try:
        return fdi.artifact_path(directory, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/fdi/tasks/{task_id}/images/{image_name}")
def fdi_task_image(task_id: str, image_name: str):
    if not image_name.endswith((".svg", ".png")):
        raise HTTPException(status_code=404, detail="FDI 图片不存在。")
    path = _fdi_artifact_path(task_id, image_name, preview=True)
    # Read one complete revision before responding; a live SVG can be replaced
    # between FileResponse's stat and streaming operations on Windows.
    return Response(path.read_bytes(), media_type="image/svg+xml" if path.suffix == ".svg" else "image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/fdi/tasks/{task_id}/files/{file_name}")
def fdi_artifact(task_id: str, file_name: str):
    path = _fdi_artifact_path(task_id, file_name)
    return FileResponse(path, filename=path.name)


@app.get("/api/replay/defaults")
def replay_defaults():
    return replay.default_config()


@app.post("/api/replay/tasks")
def create_replay_task(request: ReplayTaskRequest):
    directory = replay.new_run_dir()
    config = request.model_dump()
    cancelled = Event()
    task = task_manager.submit("重放攻击下的自适应控制仿真",
                               lambda log: replay.run(config, log, run_dir=directory, cancelled=cancelled.is_set),
                               artifact_dir=directory, progress_path=directory / "progress.json", cancel_event=cancelled)
    return {"task_id": task.id, "status": task.status}


def _replay_artifact_path(task_id, name, *, preview=False):
    if not replay.is_artifact(name):
        raise HTTPException(status_code=404, detail="重放攻击结果文件不存在。")
    task = task_manager.get(task_id)
    directory = task_manager.artifact_dir(task_id)
    if task is None or directory is None:
        raise HTTPException(status_code=404, detail="重放攻击任务不存在或本机服务已重启，请重新运行。")
    if task["status"] != "succeeded":
        progress = task.get("progress") or {}
        if not (preview and name.endswith(".svg") and progress.get("module") == "replay"
                and progress.get("outputs", {}).get(Path(name).stem)):
            raise HTTPException(status_code=409, detail="仿真尚未成功完成。")
    elif (task.get("result") or {}).get("module") != "replay":
        raise HTTPException(status_code=404, detail="该任务没有 重放攻击结果。")
    try:
        return replay.artifact_path(directory, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/replay/tasks/{task_id}/images/{image_name}")
def replay_task_image(task_id: str, image_name: str):
    if not image_name.endswith((".svg", ".png")):
        raise HTTPException(status_code=404, detail="重放攻击图片不存在。")
    path = _replay_artifact_path(task_id, image_name, preview=True)
    return Response(path.read_bytes(), media_type="image/svg+xml" if path.suffix == ".svg" else "image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/replay/tasks/{task_id}/files/{file_name}")
def replay_artifact(task_id: str, file_name: str):
    path = _replay_artifact_path(task_id, file_name)
    return FileResponse(path, filename=path.name)


@app.get("/api/self-healing/defaults")
def self_healing_defaults():
    return self_healing.default_config()


@app.post("/api/self-healing/tasks")
def create_self_healing_task(request: SelfHealingTaskRequest):
    config = self_healing.validate_config(request.config)
    directory = self_healing.new_run_dir()
    cancelled = Event()
    task = task_manager.submit("多变量过程自愈控制仿真",
                               lambda log: self_healing.run(config, log, run_dir=directory, cancelled=cancelled.is_set),
                               artifact_dir=directory, progress_path=directory / "progress.json", cancel_event=cancelled)
    return {"task_id": task.id, "status": task.status}


def _self_healing_artifact_path(task_id, name, *, preview=False):
    if not self_healing.is_artifact(name):
        raise HTTPException(status_code=404, detail="自愈控制结果文件不存在。")
    task = task_manager.get(task_id)
    directory = task_manager.artifact_dir(task_id)
    if task is None or directory is None:
        raise HTTPException(status_code=404, detail="自愈控制任务不存在，请重新运行。")
    if task["status"] != "succeeded":
        progress = task.get("progress") or {}
        if not (preview and name.endswith(".svg") and progress.get("module") == "self_healing"
                and progress.get("outputs", {}).get(Path(name).stem)):
            raise HTTPException(status_code=409, detail="仿真尚未成功完成。")
    elif (task.get("result") or {}).get("module") != "self_healing":
        raise HTTPException(status_code=404, detail="该任务没有自愈控制结果。")
    try:
        return self_healing.artifact_path(directory, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/self-healing/tasks/{task_id}/images/{image_name}")
def self_healing_task_image(task_id: str, image_name: str):
    if not image_name.endswith((".svg", ".png")):
        raise HTTPException(status_code=404, detail="自愈控制图片不存在。")
    path = _self_healing_artifact_path(task_id, image_name, preview=True)
    return Response(path.read_bytes(), media_type="image/svg+xml" if path.suffix == ".svg" else "image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/self-healing/tasks/{task_id}/files/{file_name}")
def self_healing_artifact(task_id: str, file_name: str):
    path = _self_healing_artifact_path(task_id, file_name)
    return FileResponse(path, filename=path.name)


@app.get("/{requested_path:path}", include_in_schema=False)
def frontend(requested_path: str):
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
