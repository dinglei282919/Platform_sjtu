"""A clock and journal for each stand-alone PI experiment."""
from collections import deque
from dataclasses import asdict, dataclass
from threading import Condition, Thread
import time
from uuid import uuid4

from .comparison_service import ComparisonService, sample_bands
from .journal import Journal
from .session_lifecycle import lease_action
from .replay import ARMS, CASES, DEMO_CASES, DT, END, FIXTURE, RECOVERY_STEPS, START, T_REF, Trial, scores
from .replay_display import CASE_GUIDES, CHARTS, GUIDE, HEAT_CHART, OBSERVATIONS, QUANTITIES, SENSORS, display, trend


@dataclass(frozen=True)
class Config:
    scenario: str = "constant"
    speed: float = 30.

    @classmethod
    def parse(cls, fields=None):
        cfg = cls(**(fields or {}))
        if cfg.scenario not in DEMO_CASES:
            raise ValueError("请选择已验证的演示工况")
        if type(cfg.speed) not in (int, float) or not 1 <= cfg.speed <= 1200:
            raise ValueError("推进倍率须在 1 至 1200 之间")
        return cfg


def options():
    return dict(scenarios=DEMO_CASES, groups=ARMS, sample_min=DT, reference=T_REF,
                start=START, stop=START+len(FIXTURE), end=END, defaults=asdict(Config()), metrics=QUANTITIES,
                charts=CHARTS, sensors=SENSORS, observations=OBSERVATIONS, heat_chart=HEAT_CHART,
                guide=GUIDE, case_guides=CASE_GUIDES)


class ReplaySession:
    def __init__(self, root, config, *, lease=False, auto_start=True):
        self.id, self.owner = uuid4().hex, uuid4().hex
        self.root, self.lock = root, Condition()
        self.leased, self.closed, self.connections = lease, False, 0
        self.last_seen = time.monotonic()
        self.receipts, self.archives = {}, {}
        self._new(config, auto_start)
        self.worker = Thread(target=self._loop, daemon=True, name="cstr-replay")
        self.worker.start()

    def _new(self, config, start):
        run, engine = uuid4().hex, Trial(config.scenario, manual=True)
        log = Journal(self.root, run)
        if hasattr(self, "log"):
            self.log.close()
            self.archives[self.run] = (self.config, self.last_published, self.log, self.completed_scores)
        self.run, self.config, self.log, self.engine = run, config, log, engine
        self.status = "running" if start else "paused"
        self.error, self.lease_paused, self.due = "", False, None
        self.recent, self.completed_scores = deque(maxlen=120), {}
        self.pending, self.next_scenario = None, None

    @property
    def last_published(self):
        return self.recent[-1]["seq"] if self.recent else -1

    def _tick(self):
        action, scenario = None, None
        if self.last_published >= START-1 and self.pending:
            if self.pending == "begin" and self.recent[-1]["phase"] == "active":
                action = "finish"  # Publish one current pair before a new segment.
            else:
                action, scenario = self.pending, self.next_scenario
                self.pending, self.next_scenario = None, None
        row = self.engine.advance(action, scenario=scenario)
        row["pending"] = self.pending
        row["scheduled"] = self.pending == "begin" or row["phase"] == "active"
        self.log.append("sample", row)
        self.recent.append(row)
        stop = row["stop_seq"]
        if stop is not None and row["seq"] == stop + RECOVERY_STEPS:
            start = row["start_seq"]
            self.completed_scores[start] = scores(self.recent, self.engine.case, start=start, stop=stop, end=row["seq"])

    def _loop(self):
        try:
            with self.lock:
                while not self.closed:
                    now = time.monotonic()
                    idle = now - self.last_seen
                    action = lease_action(self.leased, self.connections, idle, self.status)
                    if action == "expire":
                        self.closed = True
                        break
                    if action == "pause":
                        self.status, self.lease_paused = "paused", True
                    period = DT * 60 / self.config.speed
                    warming = self.last_published < START-1 and not self.error
                    if self.status == "running" or warming:
                        if warming or self.due is None or now >= self.due:
                            try:
                                self._tick()
                            except Exception as exc:
                                self.error, self.status = str(exc), "paused"
                            self.due = now + period if warming else max(time.monotonic(), (self.due or now) + period)
                            if warming:
                                self.lock.wait(.0001)
                            continue
                        self.lock.wait(min(1., self.due - now))
                    else:
                        self.lock.wait(1. if self.leased else None)
        finally:
            self.log.close()

    def owner_connected(self):
        with self.lock:
            if self.closed:
                raise KeyError("会话已释放")
            self.connections += 1
            self.last_seen = time.monotonic()
            if self.lease_paused:
                self.status, self.lease_paused, self.due = "running", False, None
            self.lock.notify_all()

    def owner_disconnected(self):
        with self.lock:
            self.connections = max(0, self.connections-1)
            self.last_seen = time.monotonic()

    def command(self, kind, key, run, settings=None):
        with self.lock:
            if self.closed:
                raise KeyError("会话已释放")
            if not isinstance(key, str) or not 1 <= len(key) <= 128:
                raise ValueError("需要有效指令标识")
            request = (run, kind, settings)
            if key in self.receipts:
                old, reply = self.receipts[key]
                if old != request:
                    raise ValueError("指令标识已用于其他操作")
                return reply
            if run != self.run:
                raise ValueError("运行已复位，请刷新后重试")
            if settings is not None and kind not in ("reset", "begin"):
                raise ValueError("此操作不接受参数")
            if kind == "reset":
                self._new(Config.parse({**asdict(self.config), **(settings or {})}), True)
            elif kind == "begin" and not self.error and not self.pending:
                fields = settings or {}
                scenario = fields.get("scenario", self.engine.scenario)
                if set(fields) - {"scenario"} or not isinstance(scenario, str) or scenario not in DEMO_CASES:
                    raise ValueError("请选择已验证的热负荷工况")
                self.pending, self.next_scenario = "begin", scenario
            elif kind == "cancel" and self.pending == "begin":
                self.pending, self.next_scenario = None, None
            elif kind == "finish" and not self.error and not self.pending and self.recent and self.recent[-1]["phase"] == "active":
                self.pending = "finish"
            elif kind == "pause":
                self.status = "paused"
            elif kind == "resume" and not self.error:
                self.status, self.due = "running", None
            else:
                raise ValueError("当前状态无法执行此操作")
            self.last_seen, self.lease_paused = time.monotonic(), False
            reply = dict(run=self.run, status=self.status, pending=self.pending)
            self.receipts[key] = (request, reply)
            self.lock.notify_all()
            return reply

    def frame(self, run=None, at=None, owner=None, arm="guard"):
        with self.lock:
            if self.closed:
                raise KeyError("会话已释放")
            if arm not in ARMS:
                raise ValueError("未知对照分支")
            if owner == self.owner:
                self.last_seen = time.monotonic()
            selected = run or self.run
            current = selected == self.run
            config, last, log, completed = (
                (self.config, self.last_published, self.log, self.completed_scores)
                if current else self.archives[selected])
            if at is not None and not 0 <= at <= last:
                raise ValueError("所选采样点超出运行记录")
            index = last if at is None else at
            first = max(0, index - 119)
            rows = None
            if current and (not self.recent or first >= self.recent[0]["seq"]):
                rows = [row for row in self.recent if first <= row["seq"] <= index]
            barrier = log.barrier() if rows is None and index >= 0 else None
            following = current and at is None
            pending = self.pending if following else None
            header = dict(run=selected, live_run=self.run, status=self.status, error=self.error,
                          live_seq=self.last_published, last=last, config=asdict(config), live_config=asdict(self.config),
                          archives=[dict(run=k, scenario=v[0].scenario, last=v[1]) for k,v in self.archives.items()])
        if rows is None:
            rows = log.samples_between(barrier, first, index) if barrier else []
            if rows:
                rows[-1].pop("kind", None)
        point = rows[-1] if rows else None
        available_arms = list(point["groups"] if point else ARMS)
        if arm not in available_arms:
            arm = "guard"
        if not following and point:
            pending = point["pending"]
        phase = point["phase"] if point else "healthy"
        segment = point["segment"] if point else 0
        scenario = point["scenario"] if point else config.scenario
        start = point["start_seq"] if point else None
        metrics, through = {}, None
        if start is not None:
            stop = point["stop_seq"]
            through = min(index, stop + RECOVERY_STEPS)
            metrics = completed.get(start) if through == stop + RECOVERY_STEPS else None
            if metrics is None:
                metrics = scores(rows, CASES[scenario], start=start, stop=stop, end=through)
        if pending:
            status_text = "已排队 · 下一段测温重放" if pending == "begin" else "已排队 · 结束本段重放"
            if following and header["status"] == "paused":
                status_text += "（继续仿真后生效）"
        elif phase == "active":
            status_text = f"第 {segment} 次重放 · 生效中（第 {index-start+1}/{len(FIXTURE)} 拍）"
        elif segment:
            status_text = f"第 {segment} 次重放 · " + ("恢复观察" if phase == "recovery" else "已完成")
        else:
            status_text = "正常测温 · 可注入重放"
        bands = sample_bands(rows)
        return dict(**header, point=point, arm=arm, available_arms=available_arms,
                    pending=pending, phase=phase, segment=segment, phase_label=status_text,
                    scenario=scenario, scheduled=pending == "begin" or phase == "active", metrics=metrics,
                    scored_through=through, display=display(point,arm,pending == "begin") if point else None,
                    samples=[trend(r, arm) for r in rows], bands=bands, band=bands[-1] if bands else None)

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.worker.join()


class ReplayService(ComparisonService):
    config_type = Config
    session_type = ReplaySession
