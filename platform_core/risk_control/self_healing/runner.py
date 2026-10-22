"""Platform execution, cancellation, progress and isolated result artifacts."""
import json
from pathlib import Path
import time
from uuid import uuid4

from platform_core.risk_control.control_paths import module_output_dir
from platform_core.risk_control.control_plots import SimulationPlotPublisher
from platform_core.risk_control.dnn_mpc.runner import atomic_write
from .config import PLOT_GROUPS, NOTICE, validate_config
from .reporting import save_result
from .simulation import ExperimentConfig, SimulationCancelled, run_experiment

DEFAULT_OUTPUT_DIR = module_output_dir("self_healing")
DATA_FILES = ("normal.csv", "fault.csv", "scsc.csv", "ccsc.csv", "metrics.csv", "summary.json", "trajectories.npz", "result.json")
ARTIFACT_NAMES = frozenset(DATA_FILES + tuple(f"{name}.{ext}" for name, _ in PLOT_GROUPS for ext in ("svg", "png")))


def new_run_dir():
    directory = DEFAULT_OUTPUT_DIR / "runs" / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path, data):
    atomic_write(Path(path), json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def artifact_path(directory, name):
    directory = Path(directory)
    path = directory / name
    if (name not in ARTIFACT_NAMES or not (directory / "self_healing_config.json").is_file()
            or not path.is_file() or path.resolve().parent != directory.resolve()):
        raise FileNotFoundError("自愈控制结果文件不存在。")
    return path


def run(config=None, log=None, *, run_dir=None, progress=None, cancelled=None):
    config = validate_config(config or {})
    directory = Path(run_dir) if run_dir is not None else new_run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # Reusing a result directory would invalidate older task links.
    if any(directory.iterdir()):
        raise ValueError("结果目录必须为空，请创建新的运行目录。")
    log = log or (lambda message: None)
    start = time.perf_counter()

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise SimulationCancelled("自愈控制仿真已取消。")

    from .plots import figures
    publisher = SimulationPlotPublisher(directory, figures, config["duration"], check_cancelled)
    last_percent = -1

    def update(percent, message, *, force=False):
        nonlocal last_percent
        if percent == last_percent and not force:
            return
        last_percent = percent
        write_json(directory / "progress.json", {"module": "self_healing", "percent": percent, "message": message,
                   "outputs": publisher.outputs, "image_revision": publisher.image_revision})
        if progress:
            progress(percent, message)

    try:
        write_json(directory / "self_healing_config.json", config)
        check_cancelled()
        update(0, "正在初始化自愈控制仿真")
        log(NOTICE)
        result = run_experiment(ExperimentConfig(**config), log, cancelled=cancelled, on_progress=update)
        check_cancelled()
        update(86, "正在保存四组轨迹与评价指标")
        save_result(result, directory, plots=False, scenario_subdirectory=False)
        check_cancelled()
        update(90, "正在生成输入与输出对比图")
        publisher.publish(result, final=True)
        check_cancelled()
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        summary.update(module="self_healing", elapsed_seconds=time.perf_counter() - start,
                       samples=len(result["time"]), plots=[{"name": name, "title": title} for name, title in PLOT_GROUPS],
                       output_dir=str(directory), notice=NOTICE)
        # Publish success only after every artifact and callback succeeds.
        update(100, "仿真完成，四组对照和结果图已生成")
        log("自愈控制仿真完成。")
        write_json(directory / "result.json", summary)
        return summary
    except Exception as exc:
        try:
            update(0, str(exc), force=True)
        except OSError as progress_error:
            log(f"无法写入失败进度：{progress_error}")
        raise
    finally:
        publisher.close()
