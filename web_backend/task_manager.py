"""Single-worker task coordination with durable, non-destructive task records."""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import errno
import logging
import os
from pathlib import Path
import re
from threading import Event, RLock
import time
from typing import Any, Callable
from uuid import uuid4


class TaskStateUnavailable(RuntimeError):
    """A task exists but cannot currently be read or saved; this is not a 404."""


def _lock_file(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _owner_is_alive(directory):
    try:
        handle = (directory / "owner.lock").open("r+b")
    except FileNotFoundError:
        return False
    with handle:
        try:
            _lock_file(handle)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN) or getattr(exc, "winerror", None) in (32, 33):
                return True
            raise
    return False


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
    cancel_event: Event | None = field(default=None, repr=False)
    progress: dict[str, Any] = field(default_factory=lambda: {"percent": 0, "message": "等待启动"})
    persistence_error: str | None = field(default=None, repr=False)
    _lease: Any = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        current_progress = dict(self.progress)
        if self.progress_path is not None and self.status in {"running", "succeeded", "failed", "cancelled"}:
            try:
                payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    current_progress.update(payload)
                    current_progress["revision"] = self.progress_path.stat().st_mtime_ns
            except (OSError, ValueError):
                pass
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "logs": list(self.logs),
            "result": self.result,
            "error": self.error,
            "cancellable": self.cancel_event is not None,
            "cancel_requested": self.cancel_event is not None and self.cancel_event.is_set(),
            "progress": current_progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "persistence_error": self.persistence_error,
        }


class TaskManager:
    """Runs local long operations serially so CPU-heavy tasks do not collide."""

    def __init__(self, runtime_root: Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="platform-web-task")
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = RLock()
        self._runtime_root = Path(runtime_root).resolve() if runtime_root is not None else Path(__file__).resolve().parents[1] / "web_runtime" / "tasks"
        self._closed = False
        # Construction/import is read-only. Old records and artifacts are retained.

    def _persist(self, task: TaskRecord) -> None:
        payload = {"format_version": 1, "task": task.snapshot(),
                   "progress_path": str(task.progress_path) if task.progress_path else None,
                   "artifact_dir": str(task.artifact_dir) if task.artifact_dir else None}
        destination = task.work_dir / "task.json"
        temporary = task.work_dir / (uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            for attempt in range(6):
                try:
                    os.replace(temporary, destination)
                    break
                except OSError as exc:
                    if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                        raise
                    time.sleep(.02 * (attempt + 1))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("无法清理任务记录临时文件：%s", temporary)

    def _save(self, task: TaskRecord) -> None:
        previous_error = task.persistence_error
        task.persistence_error = None
        try:
            self._persist(task)
        except (OSError, ValueError, TypeError) as exc:
            task.persistence_error = f"任务记录暂时无法保存：{exc}"
            if previous_error != task.persistence_error:
                task.logs.append(task.persistence_error)
                logging.getLogger(__name__).warning("%s: %s", task.id, task.persistence_error)

    @staticmethod
    def _release(task: TaskRecord) -> None:
        if task._lease is not None:
            task._lease.close()
            task._lease = None

    def _read_task(self, task_id: str) -> TaskRecord | None:
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            return None
        directory = self._runtime_root / task_id
        try:
            # Observe ownership first: a worker publishes its terminal record before
            # releasing the lease. Reading in the reverse order can mistake a task
            # that just completed for an interrupted one.
            owner_alive = _owner_is_alive(directory)
            raw = (directory / "task.json").read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise TaskStateUnavailable("任务记录暂时无法读取，请稍后重试。") from exc
        try:
            payload = json.loads(raw)
            value = payload["task"]
            if (payload.get("format_version") != 1 or not isinstance(value, dict)
                    or value.get("id") != task_id or not isinstance(value.get("title"), str)
                    or value.get("status") not in {"queued", "running", "succeeded", "failed", "cancelled"}
                    or not isinstance(value.get("logs"), list) or not all(isinstance(log, str) for log in value["logs"])
                    or not isinstance(value.get("progress"), dict)
                    or (value.get("result") is not None and not isinstance(value["result"], dict))
                    or (value.get("error") is not None and not isinstance(value["error"], str))
                    or not all(isinstance(value.get(key), str) for key in ("created_at", "updated_at"))):
                raise ValueError("Invalid task record")
            paths = {}
            for key in ("progress_path", "artifact_dir"):
                path = payload.get(key)
                if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
                    raise ValueError("Invalid artifact path")
                paths[key] = Path(path) if path else None
            task = TaskRecord(**{key: value[key] for key in ("id", "title", "status", "logs", "result", "error", "progress", "created_at", "updated_at")},
                              work_dir=directory, **paths)
            if task.status in {"queued", "running"} and not owner_alive:
                task.status = "failed"
                task.error = "服务已停止，本任务已中断，请重新运行。已有结果文件已保留。"
                task.logs.append(task.error)
                task.progress["message"] = task.error
            return task
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise TaskStateUnavailable("任务记录暂时无法解析，请稍后重试。") from exc

    def submit(
        self,
        title: str,
        operation: Callable[[Callable[[str], None]], dict[str, Any]],
        *,
        progress_path: Path | None = None,
        artifact_dir: Path | None = None,
        cancel_event: Event | None = None,
    ) -> TaskRecord:
        task_id = uuid4().hex
        task = TaskRecord(
            id=task_id,
            title=title,
            work_dir=self._runtime_root / task_id,
            progress_path=Path(progress_path).resolve() if progress_path is not None else None,
            artifact_dir=Path(artifact_dir).resolve() if artifact_dir is not None else None,
            cancel_event=cancel_event,
        )
        with self._lock:
            if self._closed:
                raise TaskStateUnavailable("服务正在停止，请稍后重新提交。")
            try:
                task.work_dir.mkdir(parents=True, exist_ok=False)
                task._lease = (task.work_dir / "owner.lock").open("w+b")
                task._lease.write(b"0")
                task._lease.flush()
                _lock_file(task._lease)
                task.logs.append("任务已创建，等待本机工作线程执行。")
                self._persist(task)
            except (OSError, ValueError, TypeError) as exc:
                self._release(task)
                raise TaskStateUnavailable("任务记录无法保存，未启动计算，请稍后重试。") from exc
            self._tasks[task.id] = task
            try:
                self._executor.submit(self._run, task.id, operation)
            except RuntimeError as exc:
                task.status, task.error = "failed", "任务未启动，请重新提交。"
                self._save(task)
                self._release(task)
                raise TaskStateUnavailable(task.error) from exc
        return task

    def _run(self, task_id: str, operation: Callable[[Callable[[str], None]], dict[str, Any]]) -> None:
        with self._lock:
            task = self._tasks[task_id]
            if task.status == "cancelled":
                return
            task.status = "running"
            task.updated_at = datetime.now(timezone.utc).isoformat()
            task.logs.append("任务开始执行。")
            self._save(task)

        def log(message: str) -> None:
            with self._lock:
                current = self._tasks[task_id]
                current.logs.append(str(message))
                current.updated_at = datetime.now(timezone.utc).isoformat()
                self._save(current)

        try:
            result = operation(log)
        except CancelledError:
            with self._lock:
                task.status = "cancelled"
                task.logs.append("仿真已取消。")
                task.updated_at = datetime.now(timezone.utc).isoformat()
                self._save(task)
                self._release(task)
            return
        except BaseException as exc:
            # A worker's SystemExit must also finish its record and release ownership.
            with self._lock:
                task = self._tasks[task_id]
                task.status = "failed"
                task.error = str(exc)
                task.logs.append(f"任务失败：{exc}")
                task.updated_at = datetime.now(timezone.utc).isoformat()
                self._save(task)
                self._release(task)
            return

        with self._lock:
            task = self._tasks[task_id]
            task.status = "succeeded"
            task.result = result
            task.logs.append("任务完成。")
            task.updated_at = datetime.now(timezone.utc).isoformat()
            self._save(task)
            self._release(task)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and task.persistence_error:
                self._save(task)
            if task is None:
                task = self._read_task(task_id)
            return task.snapshot() if task else None

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        """Only explicitly cancellable operations receive an interruption request."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                restored = self._read_task(task_id)
                if restored is None:
                    return None
                if restored.status in {"queued", "running"}:
                    raise ValueError("任务正在另一服务实例中执行，请在原服务中取消。")
                return restored.snapshot()
            if task.cancel_event is None:
                raise ValueError("该任务不支持取消。")
            if task.status in {"queued", "running"} and not task.cancel_event.is_set():
                task.cancel_event.set()
                task.logs.append("已请求取消仿真，等待安全停止。")
                if task.status == "queued":
                    task.status = "cancelled"
                    task.progress["message"] = "仿真已取消。"
                task.updated_at = datetime.now(timezone.utc).isoformat()
                self._save(task)
                if task.status == "cancelled":
                    self._release(task)
            return task.snapshot()

    def artifact_dir(self, task_id: str) -> Path | None:
        with self._lock:
            task = self._tasks.get(task_id) or self._read_task(task_id)
            return task.artifact_dir if task else None

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for task in self._tasks.values():
                if task.status == "queued":
                    task.status = "cancelled"
                    task.progress["message"] = "服务停止，排队任务已取消，请重新运行。"
                    task.logs.append(task.progress["message"])
                    task.updated_at = datetime.now(timezone.utc).isoformat()
                    if task.cancel_event is not None:
                        task.cancel_event.set()
                    self._save(task)
                    self._release(task)
                elif task.status == "running" and task.cancel_event is not None:
                    task.cancel_event.set()
                    task.logs.append("服务停止，已请求安全取消仿真。")
                    self._save(task)
        self._executor.shutdown(wait=False, cancel_futures=True)
