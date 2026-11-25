"""Live module-3 sessions. Only the worker advances the paired branches."""
from collections import deque
from dataclasses import asdict
from threading import Condition, Thread
import time
from uuid import uuid4

import numpy as np

from . import cstr
from .comparison_service import ComparisonService, sample_bands
from .fdi import BOOK, DEMO_VIEWS, FIXED_PRIMARY_WEIGHT, GROUPS, RECOVERY_STEPS, Comparison, Config
from .fdi_control import DT, Gate, LoopSettings
from .fdi_metrics import Scores
from .journal import Journal
from .session_lifecycle import lease_action


class FdiSession:
    def __init__(self, root, config, *, lease=False, auto_start=True):
        self.id, self.owner = uuid4().hex, uuid4().hex
        self.root, self.lock = root, Condition()
        self.closed, self.leased = False, lease
        self.connections, self.last_seen = 0, time.monotonic()
        self.archives, self.receipts = {}, {}
        self._new_run(config, auto_start)
        self.worker = Thread(target=self._loop, daemon=True, name="cstr-fdi")
        self.worker.start()

    def _new_run(self, config, start):
        run = uuid4().hex
        log = Journal(self.root, run)
        try:
            log.append("event", dict(event_id=1, event="created", seq=0, t=0., config=asdict(config),
                                     casebook=BOOK["version"], plant_version=cstr.PLANT_VERSION))
            if hasattr(self, "log"):
                if not self.finalized:
                    self._event("run_reset")
                    self.status = "ended"
                    self._finalize()
                segment = self._segment_summary()
                self.past_segments[segment["segment"]] = segment, list(self.recent)
                self.archives[self.run] = (self._summary(), self.log, self.event_id, self.past_segments)
        except Exception:
            log.close()
            raise
        self.run, self.log, self.config = run, log, config
        self.engine, self.seq, self.event_id = None, -1, 1
        self.status = "running" if start else "ready"
        self.finalized = self.failed = self.lease_paused = False
        self.error, self.pending, self.preview = "", None, True
        self.pending_scenario = None
        self.past_segments = {}
        self.samples, self.recent = deque(maxlen=240), deque(maxlen=64)
        self.scores = {key: Scores() for key in GROUPS[config.side]}
        self.cycles, self.overruns, self.init_ms = deque(maxlen=1000), 0, None
        self.wall_total_ms = 0.
        self.next_due = self.last_tick = None

    def _event(self, name, **fields):
        seq = max(0, self.seq)
        item = dict(event_id=self.event_id+1, event=name, seq=seq, t=round(seq*DT, 10), **fields)
        self.log.append("event", item)
        self.recent.append(item)
        self.event_id += 1

    def _finalize(self):
        if not self.finalized:
            self.log.close()
            self.finalized = True

    def _fail(self, exc):
        self.status, self.error, self.failed = "paused", str(exc), True
        if not self.finalized:
            try:
                self._event("failed", reason=self.error)
            except (RuntimeError, OSError):
                pass

    def _publish(self, row, events, cycle_ms, wall_ms):
        row.update(run=self.run, preview=self.pending == "begin" or (self.preview and row["segment"] == 0),
                   cycle_ms=cycle_ms, wall_ms=wall_ms)
        self.log.append("sample", row)
        self.samples.append(row)
        self.seq = row["seq"]
        for name, score in self.scores.items():
            score.record(row["groups"][name], row, self.engine.stop)
        for event in events:
            self._event(event.pop("event"), **event)
        if self.seq > 0:
            self.cycles.append(cycle_ms)
            self.wall_total_ms += wall_ms
            self.overruns += int(cycle_ms > 60000*DT/self.config.speed)
        if row["evaluation_end"]:
            self._event("horizon_reached", segment=row["segment"])
        if not any(group["valid"] for group in row["groups"].values()):
            self._fail(ValueError("所有对照分支均已暂停，请查看反馈依据并复位"))

    def _advance(self):
        start = time.perf_counter()
        action = self.pending
        # For a repeat inside an active segment, expose one genuine nominal interval
        # before applying the next preset. Keep the new request queued across that interval.
        separating = action == "begin" and self.samples[-1]["phase"] == "active"
        previous = None
        if separating:
            action = "finish"
        elif action == "begin":
            previous = self._segment_summary(interrupted=True)
        row, events = self.engine.advance(action, scenario=self.pending_scenario)
        if previous is not None:
            self.past_segments[previous["segment"]] = previous, list(self.recent)
            if previous["segment"]:
                last = self.samples[-1]["groups"]
                self.scores = {arm: Scores(point if point["valid"] else None) for arm, point in last.items()}
        now = time.perf_counter()
        if not separating:
            self.pending = self.pending_scenario = None
        self._publish(row, events, 1000*(now-start), 1000*(now-self.last_tick))
        self.last_tick = now

    def _loop(self):
        try:
            while True:
                with self.lock:
                    if self.closed:
                        return
                    idle = time.monotonic()-self.last_seen
                    action = lease_action(self.leased, self.connections, idle, self.status)
                    if action == "expire":
                        if not self.finalized:
                            self._event("lease_expired")
                        self.closed = True
                        return
                    if action == "pause":
                        self._event("lease_paused")
                        self.status, self.lease_paused = "paused", True
                    if self.engine is None and not self.failed and not self.finalized:
                        run, config = self.run, self.config
                    elif self.status == "running":
                        now = time.perf_counter()
                        period = 60*DT/self.config.speed
                        if self.next_due is None:
                            self.next_due, self.last_tick = now+period, now
                        if now < self.next_due:
                            self.lock.wait(self.next_due-now)
                            continue
                        try:
                            self._advance()
                        except Exception as exc:
                            self._fail(exc)
                        self.next_due = max(self.next_due+period, time.perf_counter())
                        continue
                    else:
                        self.lock.wait(1. if self.leased else None)
                        continue
                # Import/compile and calculate boundary zero outside the UI/command lock.
                started = time.perf_counter()
                try:
                    engine = Comparison(config, continuous=True)
                    row, events = engine.advance()
                    with self.lock:
                        if self.run == run and not self.closed and not self.finalized:
                            self.engine, self.init_ms = engine, 1000*(time.perf_counter()-started)
                            self._publish(row, events, 0., 0.)
                except Exception as exc:
                    with self.lock:
                        if self.run == run and not self.finalized:
                            self._fail(exc)
        finally:
            self.log.close()

    def owner_connected(self):
        with self.lock:
            if self.closed:
                raise KeyError("会话已释放")
            self.connections += 1
            self.last_seen = time.monotonic()
            if self.lease_paused:
                self.status, self.lease_paused = "running", False
                self.next_due = None
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
            if key in self.receipts:
                request, reply = self.receipts[key]
                if request != (run, kind, settings):
                    raise ValueError("指令标识已用于其他操作")
                return reply
            if self.run != run:
                raise ValueError("运行已复位，请刷新后重试")
            if settings is not None and kind not in ("reset", "begin"):
                raise ValueError("配置变更需要应用并复位")
            self._apply(kind, settings, key)
            self.last_seen, self.lease_paused = time.monotonic(), False
            reply = dict(run=self.run, seq=self.seq, status=self.status, pending=self.pending, key=key)
            self.receipts[key] = ((run, kind, settings), reply)
            self.lock.notify_all()
            return reply

    def _apply(self, kind, settings, key):
        if kind == "reset":
            cfg = Config.parse({**asdict(self.config), **(settings or {})})
            if cfg.side != self.config.side:
                raise ValueError("请在对应端别的独立试验中运行")
            self._new_run(cfg, True)
            return
        if self.finalized:
            raise ValueError("本轮已结束，请复位后开始")
        if kind == "begin":
            if self.failed:
                raise ValueError("本轮计算已中断，请复位后开始")
            if self.pending is not None:
                raise ValueError("已有样本操作在等待采样边界")
            scenario = self.engine.scenario if self.engine else self.config.scenario
            if settings is not None:
                presets = DEMO_VIEWS[self.config.side]["presets"]
                if (not isinstance(settings, dict) or set(settings) != {"scenario"} or
                        not isinstance(settings["scenario"], str) or settings["scenario"] not in presets or
                        self.config.scenario not in presets):
                    raise ValueError("注入时只能选择本端的预置演示样本，其他配置需复位")
                scenario = settings["scenario"]
            self._event("begin_queued", key=key, scenario=scenario)
            self.pending, self.preview = "begin", True
            self.pending_scenario = scenario
        elif kind == "finish":
            if self.pending == "begin":
                self._event("segment_cancelled", key=key)
                self.pending, self.pending_scenario, self.preview = None, None, False
            elif self.samples and self.samples[-1]["phase"] == "active" and self.pending is None:
                self._event("finish_queued", key=key)
                self.pending = "finish"
            else:
                raise ValueError("当前没有可结束的样本段")
        elif kind in ("start", "resume", "pause", "end"):
            if self.failed and kind in ("start", "resume"):
                raise ValueError("本轮计算已中断，请复位后开始")
            self._event(kind, key=key)
            self.status = {"pause": "paused", "end": "ended"}.get(kind, "running")
            self.next_due = None
            if kind == "end":
                self.pending = self.pending_scenario = None
                self._finalize()
        else:
            raise ValueError("未知试验指令")

    def _segment_summary(self, *, interrupted=False):
        engine = self.engine
        segment = engine.segment if engine else 0
        stop = engine.stop if engine else None
        completed = stop is not None and self.seq >= stop + RECOVERY_STEPS
        state = "in_progress"
        if not segment:
            state = "baseline"
        elif completed:
            state = "complete"
        elif interrupted:
            state = "interrupted"
        elif self.finalized or self.failed:
            state = "ended"
        return dict(segment=segment, scenario=engine.scenario if engine else self.config.scenario,
                    start_seq=engine.start if engine else None, scheduled_stop_seq=stop,
                    stop_seq=stop if stop is not None and self.seq >= stop else None,
                    response_seq=engine.response_seq if engine else None, score_status=state,
                    score_seq=stop + RECOVERY_STEPS if completed else self.seq,
                    metrics={key: score.report() for key, score in self.scores.items()})

    def _summary(self):
        engine = self.engine
        return dict(id=self.id, run=self.run, config=asdict(self.config), status=self.status,
                    error=self.error, seq=self.seq, pending=self.pending, preview=self.preview,
                    pending_scenario=self.pending_scenario, **self._segment_summary(), recent=list(self.recent),
                    initial=engine.initial.tolist() if engine else None,
                    model_version=engine.model_version if engine else None,
                    timing=dict(initialization_ms=self.init_ms, budget_ms=60000*DT/self.config.speed,
                                p95_ms=float(np.percentile(self.cycles, 95)) if self.cycles else None,
                                effective_speed=60000*DT*self.seq/self.wall_total_ms if self.wall_total_ms else None,
                                intervals=max(0, self.seq), overruns=self.overruns))

    def snapshot(self, after=-1, owner=None, run=None):
        with self.lock:
            if owner == self.owner:
                self.last_seen = time.monotonic()
            resync = run not in (None, self.run) or after > self.seq or (
                bool(self.samples) and after < self.samples[0]["seq"]-1)
            return dict(**self._summary(), resync=resync, latest=self.samples[-1] if self.samples else None,
                        bands=sample_bands(self.samples),
                        samples=[row for row in self.samples if resync or row["seq"] > after],
                        archives=[dict(run=k, seq=v[0]["seq"], config=v[0]["config"]) for k,v in self.archives.items()])

    def _recording(self, run):
        if run in (None, self.run):
            return self._summary(), self.log, self.event_id, self.past_segments
        if run not in self.archives:
            raise KeyError("找不到本会话的运行记录")
        return self.archives[run]

    def window(self, run, first, count):
        with self.lock:
            meta, log, _, past = self._recording(run)
            if first < 0 or first > meta["seq"] or not 1 <= count <= 240:
                raise ValueError("采样窗口无效")
            last = min(meta["seq"], first+count-1)
            barrier = log.barrier()
        rows = log.samples_between(barrier, first, last)
        segment = rows[-1].get("segment", meta["segment"])
        if segment != meta["segment"]:
            summary, events = past[segment]
            meta = {**meta, **summary, "recent": events}
        return dict(**meta, latest=rows[-1], samples=rows, bands=sample_bands(rows))

    def export(self, run=None):
        with self.lock:
            meta, log, number, past = self._recording(run)
            segments = [summary for index, (summary, _) in past.items() if index]
            if meta["segment"] and meta["segment"] not in past:
                segments.append(self._segment_summary())
            barrier = log.barrier()
        initial_scenario = meta["config"]["scenario"]
        scenarios = dict.fromkeys([initial_scenario, *(item["scenario"] for item in segments)])
        return dict(**meta, segments=segments, casebook=BOOK["version"], case=BOOK["cases"][initial_scenario],
                    cases={scenario: BOOK["cases"][scenario] for scenario in scenarios},
                    plant_version=cstr.PLANT_VERSION, sample_min=DT, integration_min=cstr.INTEGRATION_MIN,
                    flow_limits=cstr.FLOW_LIMITS, gate=asdict(Gate()), fixed_primary_weight=FIXED_PRIMARY_WEIGHT,
                    controller=asdict(LoopSettings()), recovery_samples=RECOVERY_STEPS,
                    records=log.records(barrier, meta["seq"], number))

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.worker.join()


class FdiService(ComparisonService):
    config_type = Config
    session_type = FdiSession
