"""A clocked worker for interactive control sessions, independent of either UI."""
from copy import deepcopy
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
import time


class LiveControlRuntime:
    """Serialize events and integration on one worker; publish completed samples.

    A session provides advance(dt, cancelled=...), apply_event(kind, value),
    and snapshot(). Polling the snapshot never advances the model.
    """

    def __init__(self, session_factory, *, interval=.1, lease_seconds=None):
        self.session = session_factory()
        self.interval = float(interval)
        if not 0 < self.interval <= 1:
            raise ValueError("在线采样周期必须在 (0,1] 秒内")
        self._commands = Queue(maxsize=64)
        self._stop = Event()
        self._wake = Event()
        self._lock = Lock()
        self._running = True
        self._error = ""
        self._message = "正常运行 · 实时闭环计算"
        self._lease_seconds = lease_seconds
        self._last_seen = time.monotonic()
        self._step_seconds = 0.
        self._sequence = 0
        self._applied = 0
        self._snapshot = {}
        self._thread = None
        self._publish()
        self.start()

    def start(self):
        if self._thread is None and not self._stop.is_set():
            self._thread = Thread(target=self._run, name="control-live", daemon=True)
            self._thread.start()

    def touch(self):
        with self._lock:
            self._last_seen = time.monotonic()

    def command(self, kind, value=None):
        if self._stop.is_set():
            raise RuntimeError("在线会话已关闭")
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        try:
            self._commands.put_nowait((sequence, kind, value))
        except Full as exc:
            raise ValueError("调节指令过于频繁，请等待当前指令生效") from exc
        self.touch()
        self._wake.set()
        return sequence

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snapshot)

    def _publish(self):
        value = self.session.snapshot()
        value.update(status="error" if self._error else "running" if self._running else "paused",
                     message=self._error or self._message,
                     step_wall_seconds=self._step_seconds, applied_sequence=self._applied)
        with self._lock:
            self._snapshot = value

    def _drain_commands(self):
        reset_clock = False
        while not self._stop.is_set():
            try:
                sequence, kind, value = self._commands.get_nowait()
            except Empty:
                break
            try:
                if kind == "pause":
                    self._running = False
                    self._message = "已暂停 · 对象与控制器状态保留"
                elif kind == "resume":
                    if self._error:
                        raise ValueError("计算异常后请先复位")
                    self._running = True
                    self._message = "正常运行 · 实时闭环计算"
                    reset_clock = True
                else:
                    self.session.apply_event(kind, value)
                    if kind == "reset":
                        self._error = ""
                        self._running = True
                        self._message = "已复位至正常运行状态"
                        reset_clock = True
                    else:
                        self._message = f"调节已在仿真时间 {self.session.t:.2f} s 生效"
                self._applied = sequence
            except (ValueError, TypeError) as exc:
                self._message = f"调节未应用：{exc}"
            except Exception as exc:
                # Invalid numerical responses to an event must be observable,
                # just like integration failures; keep the worker for reset.
                self._error = f"在线调节已停止：{exc}"
                self._running = False
            self._publish()
        return reset_clock

    def _run(self):
        next_tick = time.monotonic() + self.interval
        try:
            while not self._stop.is_set():
                self._wake.clear()
                if self._drain_commands():
                    next_tick = time.monotonic() + self.interval
                with self._lock:
                    expired = (self._lease_seconds is not None
                               and time.monotonic() - self._last_seen > self._lease_seconds)
                if expired and self._running:
                    self._running = False
                    self._message = "连接暂时中断，已自动暂停；可继续运行"
                    self._publish()
                if self._running and time.monotonic() >= next_tick:
                    started = time.monotonic()
                    try:
                        self.session.advance(self.interval, cancelled=self._stop.is_set)
                        self._step_seconds = time.monotonic() - started
                    except Exception as exc:
                        if self._stop.is_set():
                            break
                        self._error = f"在线计算已停止：{exc}"
                        self._running = False
                    self._publish()
                    # If a solve is slow, display actual simulation time. Do not
                    # skip model states or fabricate samples to catch the clock.
                    next_tick = max(next_tick + self.interval, time.monotonic())
                timeout = max(0., next_tick - time.monotonic()) if self._running else .2
                self._wake.wait(min(timeout, .2))
        finally:
            self._running = False

    def close(self, timeout=2):
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread is not current_thread():
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()
