"""Run artifacts and progress shared by the CS worker and BS TaskManager."""
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
from platform_core.control_paths import module_output_dir
from platform_core.control_plots import SimulationPlotPublisher
from .config import PLOT_GROUPS, validate_config
from .simulation import simulate, SOLVER_SETTINGS, SimulationCancelled

DEFAULT_OUTPUT_DIR = module_output_dir("fdi")
DATA_FILES = ("signals.npz", "signals.csv", "result.json")
ARTIFACT_NAMES = frozenset(DATA_FILES + tuple(f"{key}.{ext}" for key, _ in PLOT_GROUPS for ext in ("svg", "png")))


def new_run_dir():
    directory = DEFAULT_OUTPUT_DIR / "runs" / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Windows polling readers briefly prevent rename/delete. Preserve the
        # previous complete JSON while retrying only sharing/access conflicts.
        for attempt in range(11):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 10:
                    raise
                time.sleep(min(.02 * 2**attempt, .2))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning("无法删除 FDI 临时文件 %s：%s", temporary, exc)


def artifact_path(directory, name):
    if name not in ARTIFACT_NAMES:
        raise FileNotFoundError("FDI 结果文件不存在。")
    directory = Path(directory)
    if not (directory / "fdi_config.json").is_file():
        raise FileNotFoundError("该任务没有 FDI 结果。")
    path = directory / name
    if not path.is_file() or path.resolve().parent != directory.resolve():
        raise FileNotFoundError("FDI 结果文件不存在。")
    return path


def run(config=None, log=None, *, run_dir=None, progress=None, cancelled=None):
    config = validate_config(config or {})
    directory = Path(run_dir) if run_dir is not None else new_run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "fdi_config.json", config)
    log = log or (lambda message: None)
    start = time.perf_counter()

    def update(percent, message):
        write_json(directory / "progress.json", {"percent": percent, "message": message, "module": "fdi",
                                                "outputs": publisher.outputs, "image_revision": publisher.image_revision})
        if progress is not None:
            progress(percent, message)

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise SimulationCancelled("仿真已取消。")

    from .plots import figures
    publisher = SimulationPlotPublisher(directory, figures, config["sim_time"], check_cancelled)

    try:
        log("开始 FDI 闭环仿真。")
        update(0, "正在初始化仿真")
        result = simulate(config, progress=lambda t, end: update(int(85*t/end), f"仿真时间 {t:.2f} / {end:g} 秒"),
                          cancelled=cancelled, on_snapshot=publisher.preview)
        check_cancelled()
        update(87, "正在生成四组结果图")
        publisher.publish(result, final=True,
                          on_plot=lambda index: update(89 + index * 2, f"已生成 {index+1} / 4 组结果图"))
        check_cancelled()
        np.savez_compressed(directory / "signals.npz", **result)
        columns = ("t", "yd", "x1", "e", "DF", "w1", "ew1", "ew_error", "x2", "ut", "sut", "L1", "sig", "fdi")
        np.savetxt(directory / "signals.csv", np.column_stack([result[name] for name in columns]),
                   delimiter=",", header=",".join(columns), comments="", fmt="%.12g")
        t = result["t"]
        summary = {"module": "fdi", "config": config, "solver": {"name": "SciPy BDF", **SOLVER_SETTINGS},
                   "elapsed_seconds": time.perf_counter()-start, "samples": len(t),
                   "tracking_rmse": float(np.sqrt(np.trapezoid(result["e"]**2, t)/t[-1])),
                   "disturbance_rmse": float(np.sqrt(np.trapezoid(result["ew_error"]**2, t)/t[-1])),
                   "plots": [{"name": name, "title": title} for name, title in PLOT_GROUPS],
                   "initial_performance_bound": "infinite", "output_dir": str(directory)}
        write_json(directory / "result.json", summary)
        update(100, "仿真完成，四组结果图已生成")
        log(f"FDI 仿真完成，共 {len(t)} 个采样点。")
        return summary
    except Exception as exc:
        try:
            update(0, str(exc))
        except OSError as progress_error:
            log(f"无法写入失败进度：{progress_error}")
        raise
    finally:
        publisher.close()
