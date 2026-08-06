"""In-memory, single-worker task coordination for the local web application."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from threading import RLock
from typing import Any, Callable
from uuid import uuid4


@dataclass
class TaskRecord:
    id: str
    title: str
    status: str = "queued"
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    work_dir: Path | None = field(default=None, repr=False)
    progress_path: Path | None = field(default=None, repr=False)
    artifact_dir: Path | None = field(default=None, repr=False)
    progress: dict[str, Any] = field(default_factory=lambda: {"percent": 0, "message": "等待启动"})

    def snapshot(self) -> dict[str, Any]:
        current_progress = dict(self.progress)
        if self.progress_path is not None and self.status in {"running", "succeeded", "failed"}:
            try:
                payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    current_progress.update(payload)
                    current_progress["revision"] = self.progress_path.stat().st_mtime_ns
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "logs": list(self.logs),
            "result": self.result,
            "error": self.error,
            "progress": current_progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskManager:
    """Runs local long operations serially so CPU-heavy tasks do not collide."""

    def __init__(self, runtime_root: Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="platform-web-task")
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = RLock()
        self._runtime_root = runtime_root or Path(__file__).resolve().parents[1] / "web_runtime" / "tasks"
        # These are session-only artifacts. A new local-service session starts clean.
        shutil.rmtree(self._runtime_root, ignore_errors=True)
        self._runtime_root.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        title: str,
        operation: Callable[[Callable[[str], None]], dict[str, Any]],
        *,
        progress_path: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> TaskRecord:
        task_id = uuid4().hex
        task = TaskRecord(
            id=task_id,
            title=title,
            work_dir=self._runtime_root / task_id,
            progress_path=progress_path,
            artifact_dir=artifact_dir,
        )
        task.work_dir.mkdir(parents=True, exist_ok=False)
        task.logs.append("任务已创建，等待本机工作线程执行。")
        with self._lock:
            self._tasks[task.id] = task
        self._executor.submit(self._run, task.id, operation)
        return task

    def _run(self, task_id: str, operation: Callable[[Callable[[str], None]], dict[str, Any]]) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "running"
            task.updated_at = datetime.now(timezone.utc).isoformat()
            task.logs.append("任务开始执行。")

        def log(message: str) -> None:
            with self._lock:
                current = self._tasks[task_id]
                current.logs.append(str(message))
                current.updated_at = datetime.now(timezone.utc).isoformat()

        try:
            result = operation(log)
        except Exception as exc:  # pragma: no cover - exercised through API integration
            with self._lock:
                task = self._tasks[task_id]
                task.status = "failed"
                task.error = str(exc)
                task.logs.append(f"任务失败：{exc}")
                task.updated_at = datetime.now(timezone.utc).isoformat()
            return

        with self._lock:
            task = self._tasks[task_id]
            task.status = "succeeded"
            task.result = result
            task.logs.append("任务完成。")
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.snapshot() if task else None

    def artifact_dir(self, task_id: str) -> Path | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.artifact_dir if task else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self._runtime_root, ignore_errors=True)
