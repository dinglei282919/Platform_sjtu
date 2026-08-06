"""Web adapter for the existing MATLAB-runtime MTD anomaly detector."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
CORE_OUTPUT_DIR = ROOT / "gridattackpkg_core_output"
FIGURE_DIR = ROOT / "output_figures"


def _prepare_runtime_paths(mcr_root: Path) -> Path:
    runtime_dir = mcr_root / "runtime" / "win64"
    bin_dir = mcr_root / "bin" / "win64"
    extern_dir = mcr_root / "extern" / "bin" / "win64"
    dll_path = runtime_dir / "mclmcrrt24_2.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"未找到 MATLAB Runtime R2024b DLL: {dll_path}")

    current = os.environ.get("PATH", "")
    current_parts = current.split(os.pathsep) if current else []
    prepend = [str(path) for path in (runtime_dir, bin_dir, extern_dir) if str(path).lower() not in {item.lower() for item in current_parts}]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + current_parts)
    return dll_path


def _import_runtime_modules():
    candidates = [
        ROOT / "build_python",
        ROOT / ".venv_fw8_mcr914" / "Lib" / "site-packages",
    ]
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        candidates.extend([meipass, meipass / "gridattackpkg", Path(sys.executable).resolve().parent])
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        gridattackpkg = importlib.import_module("gridattackpkg")
    except ModuleNotFoundError as exc:
        checked = " | ".join(str(path) for path in candidates)
        raise ModuleNotFoundError(f"未找到模块 gridattackpkg。已尝试路径: {checked}") from exc
    return gridattackpkg, importlib.import_module("matlab")


def _run_external_python(mcr_root: Path, percent_min: float, percent_max: float, sigma1: float, sigma2: float) -> dict[str, str]:
    runner = ROOT / "run_core_plot_with_three_params.py"
    if not runner.exists():
        raise FileNotFoundError(f"未找到外部运行脚本: {runner}")
    candidates = [
        ROOT / ".venv_fw8_mcr914" / "python.exe",
        Path(sys.executable).resolve().parent / ".venv_fw8_mcr914" / "python.exe",
    ]
    python_exe = next((path for path in candidates if path.exists()), None)
    if python_exe is None:
        raise FileNotFoundError("未找到可用的 Python 3.10 MATLAB Runtime 兼容环境: " + " | ".join(str(path) for path in candidates))
    command = [
        str(python_exe), str(runner),
        "--percent-min", str(percent_min), "--percent-max", str(percent_max),
        "--sigma1", str(sigma1), "--sigma2", str(sigma2),
        "--mcr-root", str(mcr_root), "--core-output-dir", str(CORE_OUTPUT_DIR),
    ]
    process = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    output = "\n".join(part for part in (process.stdout, process.stderr) if part)
    output = output.replace("sigma1", "measurement_noise_runtime_value").replace("sigma2", "process_disturbance_runtime_value")
    if process.returncode != 0:
        raise RuntimeError(f"外部 Python 运行失败(退出码={process.returncode})。\n{output}")
    return {"python_exe": str(python_exe), "script": str(runner), "output": output}


def run(config: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    mcr_root = Path(str(config.get("mcr_root", r"E:\MATLAB2024")).strip())
    attack_min = float(config.get("attack_min_pct", 5.0))
    attack_max = float(config.get("attack_max_pct", 10.0))
    measurement_noise = float(config.get("measurement_noise_pct", 2.0))
    process_disturbance = float(config.get("process_disturbance_pct", 5.0))
    if attack_min > attack_max:
        raise ValueError("攻击幅度最小值不能大于最大值。")
    if not 5 <= attack_min <= 50 or not 5 <= attack_max <= 50:
        raise ValueError("随机攻击强度范围必须在 5%～50% 之间。")
    if not 1 <= measurement_noise <= 30 or not 1 <= process_disturbance <= 30:
        raise ValueError("测量噪声和过程扰动强度必须在 1%～30% 之间。")

    log("正在准备 MATLAB Runtime 和异常检测输出目录。")
    dll_path = _prepare_runtime_paths(mcr_root)
    CORE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sigma1 = measurement_noise / 10.0
    sigma2 = process_disturbance / 10.0
    percent_min = attack_min / 100.0
    percent_max = attack_max / 100.0
    started = time.perf_counter()
    handle = None
    run_mode = "inprocess"
    external_info = None
    try:
        log("正在运行移动目标防御异常检测核心算法。")
        try:
            gridattackpkg, matlab = _import_runtime_modules()
            overrides = {
                "percent_range": matlab.double([percent_min, percent_max]),
                "sigma1": sigma1,
                "sigma2": sigma2,
            }
            handle = gridattackpkg.initialize()
            params = handle.make_default_params(overrides)
            core_out = handle.run_core(params, str(CORE_OUTPUT_DIR))
            handle.plot_results_from_core(core_out, nargout=0)
            output_type = type(core_out).__name__
        except Exception as inner_exc:
            message = str(inner_exc)
            if ("not supported" in message and "Python" in message) or "Python 3.11" in message:
                run_mode = "external_python"
                external_info = _run_external_python(mcr_root, percent_min, percent_max, sigma1, sigma2)
                output_type = "external_runner"
            else:
                raise
    finally:
        if handle is not None:
            try:
                handle.terminate()
            except Exception:
                pass

    stage2_file = CORE_OUTPUT_DIR / "stage2_results.mat"
    topology_png = FIGURE_DIR / "topology.png"
    detection_png = FIGURE_DIR / "detection_probability.png"
    missing = [str(path) for path in (stage2_file, topology_png, detection_png) if not path.exists()]
    if missing:
        raise RuntimeError(f"输出文件缺失: {missing}")
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mcr_root": str(mcr_root),
        "mcr_dll": str(dll_path),
        "overrides": {"percent_range": [percent_min, percent_max], "sigma1": sigma1, "sigma2": sigma2},
        "attack_strength_percent_range": [attack_min, attack_max],
        "measurement_noise_percent": measurement_noise,
        "process_disturbance_percent": process_disturbance,
        "outputs": {
            "stage2_results_mat": str(stage2_file),
            "topology_png": str(topology_png),
            "detection_probability_png": str(detection_png),
        },
        "run_mode": run_mode,
        "run_core_output_type": output_type,
        "elapsed_seconds": round(elapsed, 2),
        "image_revision": datetime.now().timestamp(),
    }
    if external_info is not None:
        result["external_runner"] = external_info
    log(f"异常检测完成，用时 {result['elapsed_seconds']:.2f} 秒。")
    return result
