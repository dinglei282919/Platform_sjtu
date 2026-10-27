"""Chemical reactor adapter for the platform background task manager."""
from concurrent.futures import CancelledError
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import json
import time
from threading import Event, Lock, Thread
from platform_core.risk_control.control_paths import module_output_dir
from platform_core.risk_control.reactor.core import Config, SimulationCancelled, simulate
from platform_core.risk_control.reactor.reporting import export_result, load_result
from platform_core.risk_control.reactor.presentation import build_diagram_series
from platform_core.risk_control.live_runtime import LiveControlRuntime
from platform_core.risk_control.reactor.live import ReactorLiveSession

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = frozenset(("trajectory.csv", "trajectory.npz", "summary.json", "results.png"))


class ReactorLiveRegistry:
    """In-memory sessions with a short polling lease and bounded retention."""

    def __init__(self, *, lease_seconds=5., retention_seconds=120., limit=8):
        self.lease_seconds = lease_seconds
        self.retention_seconds = retention_seconds
        self.limit = limit
        self._sessions = {}
        self._lock = Lock()
        self._stop = Event()
        self._reaper = None

    def create(self):
        with self._lock:
            if len(self._sessions) >= self.limit:
                raise ValueError("在线会话已达到上限，请关闭闲置页面后重试")
            if self._reaper is None:
                self._stop.clear()
                self._reaper = Thread(target=self._expire, name="reactor-live-expiry", daemon=True)
                self._reaper.start()
            identifier = uuid4().hex
            runtime = LiveControlRuntime(ReactorLiveSession, lease_seconds=self.lease_seconds)
            self._sessions[identifier] = (runtime, time.monotonic())
        return {"id": identifier, **runtime.snapshot()}

    def get(self, identifier):
        with self._lock:
            if identifier not in self._sessions:
                raise KeyError("在线会话不存在或已过期")
            runtime, _ = self._sessions[identifier]
            self._sessions[identifier] = (runtime, time.monotonic())
            runtime.touch()
        return runtime

    def snapshot(self, identifier):
        return {"id": identifier, **self.get(identifier).snapshot()}

    def command(self, identifier, kind, value=None):
        runtime = self.get(identifier)
        sequence = runtime.command(kind, value)
        return {"id": identifier, **runtime.snapshot(), "queued_sequence": sequence}

    def delete(self, identifier):
        with self._lock:
            entry = self._sessions.pop(identifier, None)
        if entry is not None:
            entry[0].close()

    def _expire(self):
        while not self._stop.wait(1.):
            with self._lock:
                expired = [key for key, (_, seen) in self._sessions.items()
                           if time.monotonic() - seen > self.retention_seconds]
                runtimes = [self._sessions.pop(key)[0] for key in expired]
            for runtime in runtimes:
                runtime.close()

    def close(self):
        self._stop.set()
        with self._lock:
            runtimes = [entry[0] for entry in self._sessions.values()]
            self._sessions.clear()
            reaper, self._reaper = self._reaper, None
        for runtime in runtimes:
            runtime.close()
        if reaper is not None:
            reaper.join(2)


live_sessions = ReactorLiveRegistry()

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


def import_saved(source, log, *, run_dir):
    """Validate a desktop export and publish an independent Web task snapshot."""
    log("正在载入已有化学反应器仿真结果。")
    result = load_result(source)
    export_result(result, run_dir)
    return {"module": "reactor", "config": result["config"], "metrics": result["metrics"],
            "files": sorted(ARTIFACTS), "imported": True}

def trajectory(directory):
    result = load_result(directory)
    data = result["data"]
    return {**{key: values.tolist() for key, values in data.items()},
            **build_diagram_series(data)}
