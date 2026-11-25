"""Run the four self-healing comparisons on one monotonic clock and one journal."""
from collections import deque
from dataclasses import asdict
from threading import Condition, RLock, Thread
import time
from uuid import uuid4

from .cstr import FLOW_LIMITS, INLET, INTEGRATION_MIN, NOMINAL, NOMINAL_FLOW, PLANT_VERSION, SAMPLE_MIN
from .healing import (BATCH_STEPS, FAULT_PROFILE, GROUPS, LIMIT, LOWER, PI_KI, PI_KP,
                      POST_FAULT_MIN, SIGNALS, TOLERANCE, UPPER, Comparison, Config, fit_maps)
from .comparison_service import ComparisonService
from .healing_metrics import EffectHistory, Metrics
from .journal import Journal
from .session_lifecycle import lease_action


class HealingSession:
    def __init__(self, root, config, *, lease=False, auto_start=True):
        self.root = root
        self.id, self.owner = uuid4().hex, uuid4().hex
        self.lock = Condition(RLock())
        self.closed = False
        self.leased, self.connections, self.last_seen = lease, 0, time.monotonic()
        self.lease_paused = False
        self.commands = {}
        self._initialize(config, auto_start)
        self.worker = Thread(target=self._loop, daemon=True, name="cstr-self-healing")
        self.worker.start()

    def _initialize(self, config, start):
        run = uuid4().hex
        engine = Comparison(config, fault_seq=None)
        log = Journal(self.root, run)
        initial = dict(run=run, cycle_ms=0., wall_ms=0., **engine.initial())
        try:
            log.append("sample", initial)
            log.append("event", dict(event="created", event_id=1, seq=0, t=0., config=asdict(config),
                                     plant_version=PLANT_VERSION, fault_mode="manual", post_fault_min=POST_FAULT_MIN))
        except Exception:
            log.close()
            raise
        previous = getattr(self, "log", None)
        self.engine, self.config, self.run, self.log = engine, config, run, log
        self.finalized = False
        self.samples = deque([initial], maxlen=640)
        self.recent = deque(maxlen=64)
        self.event_id = 1
        self.metrics = Metrics()
        self.effect = EffectHistory(config.scenario, initial)
        self.error, self.lease_paused = "", False
        self.status = "running" if start else "ready"
        self.next_due = self.last_tick = None
        if previous is not None:
            previous.close()

    def _event(self, name, **fields):
        number = self.event_id+1
        self.log.append("event", dict(event_id=number, event=name, seq=self.engine.seq,
                                      t=self.engine.seq*SAMPLE_MIN, **fields))
        self.event_id = number

    def _finish_recording(self):
        if not self.finalized:
            self.log.close()
            self.finalized = True

    def _advance(self, wall_ms=None):
        start = time.perf_counter()
        row, proposed = self.engine.prepare()
        now = time.perf_counter()
        row.update(run=self.run, cycle_ms=1000*(now-start), wall_ms=wall_ms if wall_ms is not None else 1000*(now-self.last_tick))
        self.log.append("sample", row)
        self.engine.commit(proposed)
        self.last_tick = now
        self.samples.append(row)
        self.recent.extend(row["compensation"])
        fault_seq = self.engine.fault_seq
        self.metrics.record(row, 60000*SAMPLE_MIN/self.config.speed, fault_seq)
        self.effect.record(row, fault_seq)
        if fault_seq is not None and self.engine.seq - fault_seq >= round(POST_FAULT_MIN / SAMPLE_MIN):
            self.status = "ended"
            self._event("horizon_reached")
            self._finish_recording()

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
                    if self.status != "running":
                        self.next_due = self.last_tick = None
                        self.lock.wait(1. if self.leased else None)
                        continue
                    engine = self.engine
                    if engine.maps is not None:
                        period = 60*SAMPLE_MIN/self.config.speed
                        now = time.perf_counter()
                        if self.next_due is None:
                            self.next_due, self.last_tick = now+period, now
                        if self.next_due > now:
                            self.lock.wait(self.next_due-now)
                            continue
                        try:
                            self._advance()
                        except (ValueError, FloatingPointError, RuntimeError, OSError) as exc:
                            self._fail(exc)
                        self.next_due += period
                        # A coarse OS wake-up may span several deadlines. Compute every due
                        # interval in order. Rebase only after a long suspension or stall.
                        if self.next_due < now-max(1., 4*period):
                            self.next_due = time.perf_counter()+period
                        continue
                # Fitting and first JIT compilation take place off the UI and command locks.
                try:
                    maps = fit_maps(engine.config.seed)
                except Exception as exc:
                    with self.lock:
                        if self.engine is engine:
                            self._fail(exc)
                    continue
                with self.lock:
                    if self.engine is engine and not self.closed and not self.finalized:
                        engine.maps = maps
                        try:
                            self._event("maps_ready", version=maps[-1]["version"])
                        except (RuntimeError, OSError) as exc:
                            self._fail(exc)
        finally:
            self.log.close()

    def _fail(self, exc):
        self.status, self.error = "paused", str(exc)
        try:
            self._event("failed", reason=self.error)
        except (RuntimeError, OSError):
            pass

    def owner_connected(self):
        with self.lock:
            if self.closed:
                raise ValueError("会话已释放")
            self.connections += 1
            self.last_seen = time.monotonic()
            if self.lease_paused:
                self.lease_paused, self.status = False, "running"
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
            if key in self.commands:
                return self.commands[key]
            if not self.finalized:
                self._event("command_received", command=kind, key=key, target_run=run)
            try:
                self._apply(kind, run, settings)
            except (TypeError, ValueError) as exc:
                if not self.finalized:
                    self._event("command_rejected", command=kind, key=key, reason=str(exc))
                raise
            self.lease_paused, self.error = False, ""
            self.next_due = self.last_tick = None
            if not self.finalized:
                try:
                    self._event("command_applied", command=kind, key=key, target_run=run)
                except (RuntimeError, OSError):
                    if kind == "degrade":
                        self.engine.fault_seq = None
                    raise
            if self.status == "ended":
                self._finish_recording()
            reply = dict(run=self.run, seq=self.engine.seq, t=self.engine.seq*SAMPLE_MIN, status=self.status, key=key)
            self.commands[key] = reply
            self.lock.notify_all()
            return reply

    def _apply(self, kind, run, settings):
        if self.run != run:
            raise ValueError("运行已复位，请刷新后重试")
        if kind not in ("start", "pause", "resume", "end", "reset", "degrade"):
            raise ValueError("未知自愈指令")
        if kind != "reset" and settings is not None:
            raise ValueError("配置变更需要应用并复位")
        if kind == "reset":
            config = Config.parse(settings) if settings is not None else self.config
            self._initialize(config, True)
        elif kind == "end":
            self.status = "ended"
        elif self.status == "ended":
            raise ValueError("运行已结束，请复位后重新开始")
        elif kind == "degrade":
            if self.engine.fault_seq is not None:
                raise ValueError("本轮已施加退化，复位后可重新试验")
            self.effect.keep_latest()
            self.engine.fault_seq = self.engine.seq
        else:
            self.status = "paused" if kind == "pause" else "running"

    def snapshot(self, after=-1, owner=None, run=None):
        with self.lock:
            if owner == self.owner:
                self.last_seen = time.monotonic()
            seq = self.engine.seq
            onset = self.engine.fault_seq
            resync = run not in (None, self.run) or after > seq or after < self.samples[0]["seq"]-1
            return dict(id=self.id, run=self.run, status=self.status, error=self.error, seq=seq,
                        resync=resync, latest=self.samples[-1], config=asdict(self.config),
                        samples=[r for r in self.samples if resync or r["seq"]>after], recent=list(self.recent),
                        mapping=self.engine.maps[-1] if self.engine.maps is not None else None,
                        fault_min=None if onset is None else onset*SAMPLE_MIN,
                        effect=self.effect.snapshot(), metrics=self.metrics.summary())

    def history(self, run, first, count):
        with self.lock:
            if run != self.run or first < 0 or first > self.engine.seq or not 1 <= count <= 640:
                raise ValueError("历史窗口无效或运行已复位")
            log, barrier = self.log, self.log.barrier()
            last = min(first+count-1, self.engine.seq)
        return log.samples_between(barrier, first, last)

    def export(self):
        with self.lock:
            seq, number, log, barrier = self.engine.seq, self.event_id, self.log, self.log.barrier()
            factor, rate, floor = FAULT_PROFILE[self.config.scenario]
            onset = self.engine.fault_seq
            fault_min = None if onset is None else onset*SAMPLE_MIN
            report = dict(session=self.id, run=self.run, status=self.status, error=self.error, config=asdict(self.config),
                          plant_version=PLANT_VERSION, initial=NOMINAL.tolist(), sample_min=SAMPLE_MIN,
                          rk4_substep_min=INTEGRATION_MIN, signals=SIGNALS, groups=GROUPS, fault_min=fault_min,
                          fault_mode="manual", post_fault_min=POST_FAULT_MIN,
                          end_min=None if fault_min is None else fault_min+POST_FAULT_MIN,
                          fault=dict(factor=factor, exponential_rate=rate, floor=floor),
                          pi=dict(kp=PI_KP, ki=PI_KI, initial_integral=NOMINAL_FLOW, flow_limits=FLOW_LIMITS),
                          inlets=dict(nominal=INLET.tolist(), lower=LOWER.tolist(), upper=UPPER.tolist(), step_limit=LIMIT.tolist()),
                          output_reference=NOMINAL.tolist(), trigger_reference=list(self.config.references),
                          recovery_reference=NOMINAL.tolist(),
                          recovery_tolerance_CT=TOLERANCE.tolist(),
                          batch_samples=BATCH_STEPS, cycle_budget_ms=60000*SAMPLE_MIN/self.config.speed,
                          mapping=self.engine.maps[-1] if self.engine.maps is not None else None,
                          metrics=self.metrics.summary())
        report["records"] = log.records(barrier, seq, number)
        return report

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.worker.join()


class HealingService(ComparisonService):
    config_type = Config
    session_type = HealingSession
