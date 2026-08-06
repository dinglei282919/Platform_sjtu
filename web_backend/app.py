"""FastAPI application for the local browser/server edition of the platform."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND_DIST = ROOT / "web_frontend" / "dist"

from web_backend.task_manager import TaskManager
from web_backend.services import anomaly, cdq, classification, mpc, scoring, sdg, sil, training


task_manager = TaskManager()


class ScoreRequest(BaseModel):
    values: dict[str, float]
    weights: dict[str, float] = Field(default_factory=dict)


class SDGNodeRequest(BaseModel):
    id: str
    name: str
    type: Literal["R", "P", "C"]
    probability: float = 0.0


class SDGEdgeRequest(BaseModel):
    source: str
    target: str
    type: Literal["+", "-"] = "+"
    probability: float


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


class SILTaskRequest(BaseModel):
    m: int = 2
    n: int = 4
    lambda_fit: float = 111.11
    ti: float = 8760
    mrt: float = 8
    nsim: int = 500
    years: int = 10000
    ccf_mode: Literal["total", "partial"] = "total"
    total_beta: float = 0.1
    partial_betas: dict[str, float] = Field(default_factory=dict)


class ClassificationTaskRequest(BaseModel):
    dataset: Literal["original", "easy", "hard"] = "original"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001


class AnomalyTaskRequest(BaseModel):
    mcr_root: str = r"E:\MATLAB2024"
    attack_min_pct: float = 5.0
    attack_max_pct: float = 10.0
    measurement_noise_pct: float = 2.0
    process_disturbance_pct: float = 5.0


class TrainingTaskRequest(BaseModel):
    package_dir: str = str(training.DEFAULT_PACKAGE_DIR)
    package_name: str = training.DEFAULT_PACKAGE_NAME
    mcr_root: str = r"E:\MATLAB2024"
    output_dir: str = str(training.DEFAULT_OUTPUT_DIR)
    model_path: str = str(training.DEFAULT_OUTPUT_DIR / "process_control_nn_model.mat")
    sample_count: int = 1000
    epochs: int = 50
    hidden_layers: str = "64,64"
    dataset_path: str = ""


class MpcTaskRequest(BaseModel):
    package_dir: str = str(mpc.DEFAULT_PACKAGE_DIR)
    package_name: str = mpc.DEFAULT_PACKAGE_NAME
    mcr_root: str = r"E:\MATLAB2024"
    output_dir: str = str(mpc.DEFAULT_OUTPUT_DIR)
    model_path: str = str(mpc.DEFAULT_OUTPUT_DIR / "process_control_nn_model.mat")
    sim_time: float = 1.0
    prediction_horizon: int = 5


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


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "mode": "local-only", "host": "127.0.0.1", "frontend_built": (FRONTEND_DIST / "index.html").is_file()}


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
        {"id": "training", "name": "控制模型训练评估", "status": "available"},
        {"id": "dnn-mpc", "name": "DNN-MPC优化控制仿真", "status": "available"},
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
        return cdq.analyze(request.step, request.horizon, request.sample_index, request.cv, request.u_now, request.u_after)
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
    output_dir = Path(str(config.get("output_dir", "")).strip() or str(training.DEFAULT_OUTPUT_DIR))
    task = task_manager.submit(
        "控制模型训练评估 · DNNTrain",
        lambda log: training.run(config, log),
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
    return FileResponse(image, media_type="image/png")


@app.get("/api/training/tasks/{task_id}/images/{image_name}")
def training_task_image(task_id: str, image_name: str) -> FileResponse:
    output_dir = task_manager.artifact_dir(task_id)
    if output_dir is None:
        raise HTTPException(status_code=404, detail="训练任务不存在或本机服务已重启")
    try:
        image = training.image_path(image_name, output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image, media_type="image/png")


@app.get("/api/mpc/defaults")
def mpc_defaults() -> dict[str, Any]:
    return mpc.default_config()


@app.post("/api/mpc/tasks")
def create_mpc_task(request: MpcTaskRequest) -> dict[str, str]:
    config = request.model_dump()
    output_dir = Path(str(config.get("output_dir", "")).strip() or str(mpc.DEFAULT_OUTPUT_DIR))
    task = task_manager.submit(
        "优化控制仿真验证 · MPC simulation",
        lambda log: mpc.run(config, log),
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
    return FileResponse(image, media_type="image/png")


@app.get("/api/mpc/tasks/{task_id}/images/{image_name}")
def mpc_task_image(task_id: str, image_name: str) -> FileResponse:
    output_dir = task_manager.artifact_dir(task_id)
    if output_dir is None:
        raise HTTPException(status_code=404, detail="MPC 任务不存在或本机服务已重启")
    try:
        image = mpc.image_path(image_name, output_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(image, media_type="image/png")


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
        raise HTTPException(status_code=404, detail="任务不存在或本机服务已重启")
    return task


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
