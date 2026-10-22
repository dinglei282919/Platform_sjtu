"""Chemical reactor adapter for the platform background task manager."""
from concurrent.futures import CancelledError
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import json
from platform_core.control_paths import module_output_dir
from platform_core.reactor.core import Config, SimulationCancelled, simulate
from platform_core.reactor.reporting import export_result, load_result

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = frozenset(("trajectory.csv", "trajectory.npz", "summary.json", "results.png"))

def default_config():
    return asdict(Config())

def new_run_dir():
    path = module_output_dir("reactor") / "runs" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path

def run(config, log, *, run_dir, cancelled=None):
    directory = Path(run_dir)
    def progress(fraction):
        # TaskManager tolerates a temporarily unreadable progress file.
        (directory / "progress.json").write_text(json.dumps({
            "module": "reactor", "percent": round(100*fraction),
            "message": "计算中；工艺动画暂停，可取消"
        }), encoding="utf-8")
    log("化学反应器计算开始；使用编译数值内核，首次运行需初始化。")
    try:
        result = simulate(Config(**config), progress=progress, cancelled=cancelled)
        if cancelled and cancelled():
            raise SimulationCancelled("仿真已取消")
        export_result(result, directory)
    except SimulationCancelled as exc:
        raise CancelledError(str(exc)) from exc
    progress(1)
    return {"module": "reactor", "config": result["config"], "metrics": result["metrics"],
            "files": sorted(ARTIFACTS)}

def artifact_path(directory, name):
    path = Path(directory) / name
    if name not in ARTIFACTS or path.resolve().parent != Path(directory).resolve() or not path.is_file():
        raise FileNotFoundError("化学反应器结果文件不存在")
    return path

def trajectory(directory):
    result = load_result(directory)
    return {key: values.tolist() for key, values in result["data"].items()}
