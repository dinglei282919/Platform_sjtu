"""Web adapter for the CS process-control MPC validation module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .training import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PACKAGE_DIR,
    DEFAULT_PACKAGE_NAME,
    _package_module,
    _runtime_library,
)


ROOT = Path(__file__).resolve().parents[2]
LATEST_TRAINING_STATE_PATH = ROOT / "dnn_mpc" / "latest_training_state.json"
FIGURE_NAMES = (
    "process_control_trajectory.png",
    "control_input.png",
    "tracking_error.png",
    "cost_curve.png",
)


def default_config() -> dict[str, Any]:
    output_dir = DEFAULT_OUTPUT_DIR
    model_path = output_dir / "process_control_nn_model.mat"
    try:
        state = json.loads(LATEST_TRAINING_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            output_text = str(state.get("output_dir", "")).strip()
            model_text = str(state.get("model_mat", "")).strip()
            if output_text:
                output_dir = Path(output_text)
            if model_text:
                model_path = Path(model_text)
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "package_dir": str(DEFAULT_PACKAGE_DIR),
        "package_name": DEFAULT_PACKAGE_NAME,
        "mcr_root": r"E:\MATLAB2024",
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "sim_time": 1.0,
        "prediction_horizon": 5,
    }


def image_path(image_name: str, output_dir: Path | None = None) -> Path:
    if image_name not in FIGURE_NAMES:
        raise FileNotFoundError("MPC 仿真结果图片不存在")
    path = (output_dir or DEFAULT_OUTPUT_DIR) / image_name
    if not path.is_file():
        raise FileNotFoundError(f"MPC 仿真结果尚未生成：{image_name}")
    return path


def run(config: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    package_name = str(config.get("package_name", DEFAULT_PACKAGE_NAME)).strip() or DEFAULT_PACKAGE_NAME
    package_text = str(config.get("package_dir", "")).strip()
    package_dir = Path(package_text) if package_text else None
    mcr_text = str(config.get("mcr_root", "")).strip()
    output_text = str(config.get("output_dir", "")).strip()
    model_text = str(config.get("model_path", "")).strip()
    output_dir = Path(output_text)
    model_path = Path(model_text)
    sim_time = float(config.get("sim_time", 1.0))
    prediction_horizon = int(config.get("prediction_horizon", 5))

    if not mcr_text:
        raise ValueError("请填写 MATLAB Runtime 根目录 MCR_ROOT")
    if not output_text:
        raise ValueError("请填写输出目录")
    if not model_text or not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    if not 0.2 <= sim_time <= 20.0:
        raise ValueError("仿真时长应在 0.2 到 20 秒之间")
    if not 1 <= prediction_horizon <= 60:
        raise ValueError("预测步长应在 1 到 60 之间")

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        (output_dir / "progress.json").unlink()
    except FileNotFoundError:
        pass

    _, runtime_library = _runtime_library(Path(mcr_text))
    package_module = _package_module(package_dir, package_name)
    log(f"启动 MPC simulation：仿真时长 {sim_time:g}s，预测步长 {prediction_horizon}")
    handle = package_module.initialize()
    try:
        result_json = str(
            handle.run_process_control_mpc_validation(
                str(output_dir), str(model_path), sim_time, float(prediction_horizon)
            )
        )
    finally:
        try:
            handle.terminate()
        except Exception:  # pragma: no cover - runtime cleanup best effort
            pass

    result = json.loads(result_json)
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    figure_paths = {
        name: Path(str(outputs.get(key, output_dir / name)))
        for key, name in (
            ("trajectory_figure", "process_control_trajectory.png"),
            ("control_figure", "control_input.png"),
            ("tracking_error_figure", "tracking_error.png"),
            ("cost_figure", "cost_curve.png"),
        )
    }
    revision = max((path.stat().st_mtime_ns for path in figure_paths.values() if path.is_file()), default=0)
    log("MPC simulation 完成")
    return {
        "mode": "mpc",
        "backend": "matlab_python_package",
        "backend_detail": {
            "package_name": package_name,
            "package_dir": str(package_dir or ""),
            "mcr_root": mcr_text,
            "runtime_library": str(runtime_library),
        },
        "result": result,
        "result_json": result_json,
        "json_path": str(outputs.get("summary_json", output_dir / "mpc_validation_summary.json")),
        "outputs": outputs,
        "image_revision": revision,
        "model_path": str(outputs.get("model_mat", model_path)),
    }
