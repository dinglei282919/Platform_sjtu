"""FastAPI application for the local browser/server edition of the platform."""

from __future__ import annotations

import sys
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND_DIST = ROOT / "web_frontend" / "dist"

from web_backend.task_manager import TaskManager
from web_backend.services import anomaly, cdq, classification, mpc, path_picker, scoring, sdg, sil, training


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
    package_dir: str = str(training.DEFAULT_PACKAGE_DIR)
    package_name: str = training.DEFAULT_PACKAGE_NAME
    mcr_root: str = r"E:\MATLAB2024"
    output_dir: str = str(training.DEFAULT_OUTPUT_DIR)
    model_path: str = str(training.DEFAULT_OUTPUT_DIR / "process_control_nn_model.mat")
    sample_count: int = Field(default=1000, ge=100, le=100000)
    epochs: int = Field(default=50, ge=1, le=5000)
    hidden_layers: str = "64,64"
    dataset_path: str = ""

    @field_validator("package_name")
    @classmethod
    def validate_package_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or not all(part.isidentifier() for part in normalized.split(".")):
            raise ValueError("Python 包名格式不正确，只能使用合法的 Python 模块名。")
        return normalized

    @field_validator("package_dir")
    @classmethod
    def validate_package_dir(cls, value: str) -> str:
        normalized = value.strip()
        if normalized and not Path(normalized).is_absolute():
            raise ValueError("Python 包目录必须是本机绝对路径；也可以留空以使用当前环境中已安装的包。")
        return normalized

    @field_validator("mcr_root", "output_dir")
    @classmethod
    def validate_required_directory(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("路径不能为空。")
        if not Path(normalized).is_absolute():
            raise ValueError("路径必须是本机绝对路径。")
        return normalized

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or not Path(normalized).is_absolute():
            raise ValueError("模型文件必须使用本机绝对路径。")
        if Path(normalized).suffix.lower() != ".mat":
            raise ValueError("模型文件必须使用 .mat 扩展名。")
        return normalized

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        if not Path(normalized).is_absolute():
            raise ValueError("外部数据集必须使用本机绝对路径。")
        if Path(normalized).suffix.lower() != ".mat":
            raise ValueError("外部数据集必须是 .mat 文件。")
        return normalized

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, value: str) -> str:
        try:
            layers = [int(part.strip()) for part in value.split(",")]
        except (AttributeError, ValueError) as exc:
            raise ValueError("隐藏层规模必须是用逗号分隔的正整数，例如 64,64。") from exc
        if not layers or len(layers) > 10 or any(layer < 1 or layer > 4096 for layer in layers):
            raise ValueError("隐藏层最多设置 10 层，每层神经元数量必须是 1～4096 之间的整数。")
        return ",".join(str(layer) for layer in layers)


class MpcTaskRequest(BaseModel):
    package_dir: str = str(mpc.DEFAULT_PACKAGE_DIR)
    package_name: str = mpc.DEFAULT_PACKAGE_NAME
    mcr_root: str = r"E:\MATLAB2024"
    output_dir: str = str(mpc.DEFAULT_OUTPUT_DIR)
    model_path: str = str(mpc.DEFAULT_OUTPUT_DIR / "process_control_nn_model.mat")
    sim_time: float = Field(default=1.0, ge=0.2, le=20.0, allow_inf_nan=False)
    prediction_horizon: int = Field(default=5, ge=1, le=60)

    @field_validator("package_name")
    @classmethod
    def validate_package_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or not all(part.isidentifier() for part in normalized.split(".")):
            raise ValueError("Python 包名格式不正确，只能使用合法的 Python 模块名。")
        return normalized

    @field_validator("package_dir")
    @classmethod
    def validate_package_dir(cls, value: str) -> str:
        normalized = value.strip()
        if normalized and not Path(normalized).is_absolute():
            raise ValueError("Python 包目录必须是本机绝对路径；也可以留空以使用当前环境中已安装的包。")
        return normalized

    @field_validator("mcr_root", "output_dir")
    @classmethod
    def validate_required_directory(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("路径不能为空。")
        if not Path(normalized).is_absolute():
            raise ValueError("路径必须是本机绝对路径。")
        return normalized

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or not Path(normalized).is_absolute():
            raise ValueError("模型文件必须使用本机绝对路径。")
        if Path(normalized).suffix.lower() != ".mat":
            raise ValueError("模型文件必须使用 .mat 扩展名。")
        return normalized

class LocalPathSelectRequest(BaseModel):
    kind: Literal["directory", "open-mat", "save-mat"]
    initial_path: str = ""


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
        {"id": "training", "name": "风险场景下控制系统智能模型训练算法", "status": "available"},
        {"id": "dnn-mpc", "name": "风险场景下智能模型优化控制算法", "status": "available"},
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
    output_dir = Path(str(config.get("output_dir", "")).strip() or str(training.DEFAULT_OUTPUT_DIR))
    task = task_manager.submit(
        "风险场景下控制系统智能模型训练算法 · DNNTrain",
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
        "风险场景下智能模型优化控制算法 · MPC simulation",
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
