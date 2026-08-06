"""Web adapter for the CS process-control DNN training module."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "dnn_mpc" / "output"
DEFAULT_PACKAGE_DIR = ROOT / "dnn_mpc" / "build_python"
DEFAULT_PACKAGE_NAME = "dnnmpcpkg"
LATEST_TRAINING_STATE_PATH = ROOT / "dnn_mpc" / "latest_training_state.json"
FIGURE_NAMES = ("training_performance.png", "prediction_error.png")


def default_config() -> dict[str, Any]:
    output_dir = DEFAULT_OUTPUT_DIR
    return {
        "package_dir": str(DEFAULT_PACKAGE_DIR),
        "package_name": DEFAULT_PACKAGE_NAME,
        "mcr_root": r"E:\MATLAB2024",
        "output_dir": str(output_dir),
        "model_path": str(output_dir / "process_control_nn_model.mat"),
        "sample_count": 1000,
        "epochs": 50,
        "hidden_layers": "64,64",
        "dataset_path": "",
    }


def image_path(image_name: str, output_dir: Path | None = None) -> Path:
    if image_name not in FIGURE_NAMES:
        raise FileNotFoundError("训练结果图片不存在")
    path = (output_dir or DEFAULT_OUTPUT_DIR) / image_name
    if not path.is_file():
        raise FileNotFoundError(f"训练结果尚未生成：{image_name}")
    return path


def _runtime_library(mcr_root: Path) -> tuple[str, Path]:
    system = platform.system()
    if system == "Windows":
        arch, path_var, filename = "win64", "PATH", "mclmcrrt24_2.dll"
    elif system == "Linux":
        arch, path_var, filename = "glnxa64", "LD_LIBRARY_PATH", "libmwmclmcrrt.so.24.2"
    elif system == "Darwin":
        arch = "maca64" if platform.machine().lower() in {"arm64", "aarch64"} else "maci64"
        path_var, filename = "DYLD_LIBRARY_PATH", "libmwmclmcrrt.24.2.dylib"
    else:
        raise RuntimeError(f"不支持的操作系统：{system}")

    runtime_dir = mcr_root / "runtime" / arch
    bin_dir = mcr_root / "bin" / arch
    extern_dir = mcr_root / "extern" / "bin" / arch
    library = runtime_dir / filename
    if not library.is_file():
        raise FileNotFoundError(f"未找到 MATLAB Runtime R2024b 运行库：{library}")
    current = os.environ.get(path_var, "")
    current_parts = current.split(os.pathsep) if current else []
    prepend = [str(path) for path in (runtime_dir, bin_dir, extern_dir) if path.is_dir() and str(path).lower() not in {part.lower() for part in current_parts}]
    if prepend:
        os.environ[path_var] = os.pathsep.join(prepend + current_parts)
    return path_var, library


def _package_module(package_dir: Path | None, package_name: str):
    if package_dir is not None:
        package_root = package_dir / package_name
        if not package_root.is_dir():
            raise ModuleNotFoundError(f"未找到 MATLAB Python Package：{package_root}")
        package_text = str(package_dir)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
    if importlib.util.find_spec(package_name) is None:
        raise ModuleNotFoundError(f"未找到 MATLAB Python Package：{package_name}")
    return importlib.import_module(package_name)


def run(config: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    package_name = str(config.get("package_name", DEFAULT_PACKAGE_NAME)).strip() or DEFAULT_PACKAGE_NAME
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", package_name):
        raise ValueError("Python包名必须是有效的 Python 模块名")

    package_text = str(config.get("package_dir", "")).strip()
    package_dir = Path(package_text) if package_text else None
    mcr_text = str(config.get("mcr_root", "")).strip()
    output_text = str(config.get("output_dir", "")).strip()
    model_text = str(config.get("model_path", "")).strip()
    mcr_root = Path(mcr_text)
    output_dir = Path(output_text)
    model_path = Path(model_text)
    dataset_text = str(config.get("dataset_path", "")).strip()
    dataset_path = Path(dataset_text) if dataset_text else None
    sample_count = int(config.get("sample_count", 1000))
    epochs = int(config.get("epochs", 50))
    hidden_layers = str(config.get("hidden_layers", "64,64")).strip() or "64,64"

    if not 100 <= sample_count <= 100000:
        raise ValueError("训练样本数应在 100 到 100000 之间")
    if not 1 <= epochs <= 5000:
        raise ValueError("训练轮数应在 1 到 5000 之间")
    if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", hidden_layers):
        raise ValueError("隐藏层规模应使用逗号分隔的正整数，例如 64,64")
    if not mcr_text:
        raise ValueError("请填写 MATLAB Runtime 根目录 MCR_ROOT")
    if not output_text:
        raise ValueError("请填写输出目录")
    if dataset_path is not None and not dataset_path.is_file():
        raise FileNotFoundError(f"外部数据集文件不存在：{dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Start each run from a clean progress marker, matching the CS client.
    try:
        (output_dir / "progress.json").unlink()
    except FileNotFoundError:
        pass
    _, runtime_library = _runtime_library(mcr_root)
    package_module = _package_module(package_dir, package_name)
    log(f"Python包：{package_name}")
    log(f"输出目录：{output_dir}")
    log(f"训练样本数：{sample_count}，训练轮数：{epochs}，隐藏层规模：{hidden_layers}")
    log("启动 DNNTrain 训练模块。")

    handle = package_module.initialize()
    try:
        result_json = str(handle.run_process_control_training(str(output_dir), float(sample_count), float(epochs), hidden_layers, str(dataset_path) if dataset_path else ""))
    finally:
        try:
            handle.terminate()
        except Exception:  # pragma: no cover - runtime cleanup best effort
            pass

    result = json.loads(result_json)
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    figure_paths = {name: Path(str(outputs.get(key, output_dir / name))) for key, name in (("training_figure", "training_performance.png"), ("prediction_error_figure", "prediction_error.png"))}
    revision = max((path.stat().st_mtime_ns for path in figure_paths.values() if path.is_file()), default=0)
    model_output = str(outputs.get("model_mat", model_path))
    try:
        LATEST_TRAINING_STATE_PATH.write_text(
            json.dumps(
                {
                    "model_mat": model_output,
                    "output_dir": str(output_dir),
                    "summary_json": str(outputs.get("summary_json", output_dir / "training_summary.json")),
                    "timestamp": result.get("timestamp", ""),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    log("DNNTrain 训练模块完成。")
    return {
        "mode": "training",
        "backend": "matlab_python_package",
        "backend_detail": {"package_name": package_name, "package_dir": str(package_dir or ""), "mcr_root": str(mcr_root), "runtime_library": str(runtime_library)},
        "result": result,
        "result_json": result_json,
        "json_path": str(outputs.get("summary_json", output_dir / "training_summary.json")),
        "outputs": outputs,
        "image_revision": revision,
        "model_path": model_output,
    }
