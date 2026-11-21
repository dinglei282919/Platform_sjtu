"""A single background runner owns sampling, version activation and actuator application."""
import time
from collections import deque
from dataclasses import asdict
from threading import Condition, Event, Lock, Thread
from uuid import uuid4

import numpy as np
from .cstr import DEMO_INITIAL, INLET, NOMINAL, NOMINAL_FLOW, SAMPLE_MIN, TREND_MAX_INTERVALS, apply_flow, step
from .dnn_mpc.dataset import Trajectories
from .dnn_mpc.mpc import check_reference, solve
from .journal import Journal
from .disturbance import DEFAULT_CYCLES, InletTrial


UPDATE_BATCH = 24


class Session:
    def __init__(self, root, model, settings, lease=False, auto_start=False, submit_update=None):
        self.root, self.settings = root, settings
        self.id, self.owner = uuid4().hex, uuid4().hex
        self.lock = Condition()
        self.closed = False
        self.generation = 0
        self.leased, self.last_seen = lease, time.monotonic()
        self.auto_start = auto_start
        self.owner_connections = 0
        self.lease_paused = False
        self.seen = {}
        self.initial_model = model
        self.auto_enabled = submit_update is not None
        self.update_device = "auto"
        # Serialize task registration without blocking the numerical thread on disk I/O.
        self.update_submit_lock = Lock()
        self.update_worker = None
        self._init_run()
        self.worker = Thread(target=self._run, daemon=True, name="cstr-control")
        self.worker.start()
        if submit_update is not None:
            self.update_worker = Thread(target=self._watch_updates, args=(submit_update,),
                                        daemon=True, name="cstr-update-scheduler")
            self.update_worker.start()

    def _init_run(self):
        self.run = uuid4().hex
        self.disturbance = None
        self.model = self.initial_model
        self.log = Journal(self.root, self.run)
        self.seq = self.event_number = 0
        self.state = (DEMO_INITIAL if self.auto_start else NOMINAL).copy()
        self.previous = NOMINAL_FLOW
        self.reference = self.settings.reference
        self.warm = self.pending = self.pending_ref = self.update = None
        self.trial_previous = None
        self.update_cutoff = 0
        self.last_online_result = None
        self.in_flight = False
        self.status = "running" if self.auto_start else "ready"
        self.error = None
        self.lease_paused = False
        self.samples, self.events = deque(maxlen=TREND_MAX_INTERVALS + 1), deque(maxlen=128)
        self.history = deque(maxlen=512)
        self._sample(self.previous, self.previous, self.model.version, {}, 0, self.state, 0)
        self._event("created", config=asdict(self.settings), version=self.model.version,
                    auto_update=self.auto_enabled, update_device=self.update_device, update_batch=UPDATE_BATCH)

    def _event(self, name, **values):
        number = self.event_number + 1
        record = dict(event=name, event_id=number, seq=self.seq, t=self.seq*SAMPLE_MIN, **values)
        self.log.append("event", record)
        self.event_number = number
        self.events.append(record)

    def _sample(self, command, applied, version, diagnostics, elapsed, state, number, inlet=INLET):
        record = dict(run=self.run, seq=number, t=number*SAMPLE_MIN, C=float(state[0]), T=float(state[1]),
                      Tc=float(state[2]), Qc=applied, Qc_cmd=command, T_ref=self.reference,
                      Ci=float(inlet[0]), Ti=float(inlet[1]), Tci=float(inlet[2]), feedback="ideal",
                      model=version, cycle_ms=elapsed, **diagnostics)
        self.log.append("sample", record)
        self.samples.append(record)

    def snapshot(self, after=-1, owner=None):
        with self.lock:
            if owner == self.owner:
                self.last_seen = time.monotonic()
            resync = after < self.samples[0]["seq"]-1 or after > self.seq
            return dict(id=self.id, run=self.run, status=self.status, error=self.error, seq=self.seq,
                        model=self.model.version, initial_model=self.initial_model.version, reference=self.reference, config=asdict(
                            self.settings),
                        update=self.update["id"] if self.update else None, resync=resync,
                        samples=[s for s in self.samples if resync or s["seq"] > after],
                        latest=self.samples[-1], events=list(self.events),
                        disturbance=self.disturbance.summary(self.seq) if self.disturbance else None,
                        online_update=self._online_status())

    def owner_connected(self):
        with self.lock:
            if self.closed:
                raise ValueError("会话已释放")
            self.owner_connections += 1
            self.last_seen = time.monotonic()
            if self.lease_paused and self.auto_start:
                self.lease_paused = False
                self.status = "running"
                self._event("owner_reconnected")
                self.lock.notify_all()

    def owner_disconnected(self):
        with self.lock:
            self.owner_connections -= 1
            self.last_seen = time.monotonic()

    def sample_window(self, run, first, count):
        if first < 0 or not 1 <= count <= 512:
            raise ValueError("历史窗口范围无效")
        with self.lock:
            if run != self.run:
                raise ValueError("该运行已经复位")
            last = min(self.seq, first + count - 1)
            if first >= self.samples[0]["seq"]:
                return [r for r in self.samples if first <= r["seq"] <= last]
            log = self.log
            barrier = log.barrier()
        return log.samples_between(barrier, first, last)

    def _update_due(self):
        return (self.auto_enabled and self.status == "running" and self.update is None
                and self.pending is None and self.trial_previous is None
                and self.seq - self.update_cutoff >= UPDATE_BATCH and len(self.history) >= UPDATE_BATCH)

    def _watch_updates(self, submit):
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self.closed or self._update_due())
                if self.closed:
                    return
                run, boundary = self.run, self.seq
            try:
                submit(self, automatic=True)
            except Exception as exc:
                # A dispatch error consumes this batch, never a control interval.
                with self.lock:
                    if self.run == run and not self.closed and self.update_cutoff < boundary:
                        self.update_cutoff = boundary
                        self.last_online_result = dict(state="failed", seq=boundary, detail=str(exc))

    def configure_updates(self, enabled, device):
        with self.lock:
            if self.closed or self.status == "ended":
                raise ValueError("会话已结束")
            if (self.auto_enabled, self.update_device) != (enabled, device):
                self._event("auto_update_configured", enabled=enabled, device=device)
                self.auto_enabled, self.update_device = enabled, device
                if not enabled and self.update and self.update["automatic"]:
                    self._event("update_cancelled", job=self.update["id"], reason="自动更新已关闭")
                    self._cancel_update()
                self.lock.notify_all()
            return self._online_status()

    def _online_status(self):
        if self.update:
            phase = self.update["phase"]
        elif self.status == "ended" or self.closed:
            phase = "ended"
        elif not self.auto_enabled:
            phase = "off"
        elif self.status != "running":
            phase = "waiting_run"
        else:
            phase = "queued_to_submit" if self._update_due() else "collecting"
        return dict(enabled=self.auto_enabled, device=self.update_device, phase=phase,
                    new_intervals=min(UPDATE_BATCH, self.seq-self.update_cutoff), batch_intervals=UPDATE_BATCH,
                    automatic=self.update["automatic"] if self.update else None,
                    manual_ready=(self.status != "ended" and not self.closed and self.update is None
                                  and self.pending is None and self.trial_previous is None
                                  and len(self.history) >= UPDATE_BATCH),
                    last=self.last_online_result)

    def _cancel_update(self):
        if self.update:
            self.update["cancel"].set()
            self.update_cutoff = self.seq
            self.last_online_result = dict(state="cancelled", seq=self.seq)
        self.pending = self.update = None

    def command(self, kind, key, *, value=None, reset_model=None, settings=None, preset=None, duration_cycles=None):
        with self.lock:
            if not key or len(key) > 128:
                raise ValueError("指令需要有效标识")
            if key in self.seen:
                return self.seen[key]
            if self.closed:
                raise ValueError("会话已关闭")
            self.last_seen = time.monotonic()
            details = {}
            if kind in ("start", "resume") and self.status in ("ready", "paused"):
                self.status, self.error = "running", None
                self.lease_paused = False
            elif kind == "pause" and self.status in ("running", "ready", "paused"):
                self.lease_paused = False
                self.status = "paused"
                self.in_flight = False
                self.generation += 1
            elif kind == "reference" and self.status != "ended":
                ref = check_reference(value)
                if self.pending_ref:
                    self._event("reference_superseded", key=self.pending_ref[0])
                self.pending_ref = key, ref
            elif kind == "disturbance" and self.status != "ended":
                if self.disturbance and self.disturbance.phase in ("queued", "active", "recovery"):
                    raise ValueError("已有扰动试验正在进行")
                cycles = DEFAULT_CYCLES if duration_cycles is None else duration_cycles
                self.disturbance = InletTrial(key, preset, cycles)
                details = self.disturbance.summary(self.seq)
            elif kind == "clear_disturbance" and self.status != "ended":
                if self.disturbance is None:
                    raise ValueError("当前没有待撤除的扰动")
                self.disturbance.clear()
                details = self.disturbance.summary(self.seq)
            elif kind == "model" and self.status != "ended":
                if reset_model is None or not reset_model.metadata.get("acceptance", {}).get("passed"):
                    raise ValueError("请选择已通过闭环验证的模型")
                if self.trial_previous is not None:
                    raise ValueError("请等待当前模型完成首个控制周期")
                self._cancel_update()
                if reset_model.version != self.model.version:
                    self.update_cutoff = self.seq
                    info = dict(id=key, run=self.run, base=self.model.version, cancel=Event())
                    self._event("model_requested", key=key, version=reset_model.version)
                    self.pending = reset_model, info, self.event_number
            elif kind == "reset":
                self._cancel_update()
                self.generation += 1
                try:
                    self._event("reset", key=key)
                except (OSError, RuntimeError) as exc:
                    # The new journal records a failure to close out the previous run.
                    self.error = str(exc)
                previous_log_error = self.log.error or self.error
                self.log.close()
                if reset_model is not None:
                    self.initial_model = reset_model
                if settings is not None:
                    self.settings = settings
                self._init_run()
                if previous_log_error:
                    self._event("previous_run_error", reason=previous_log_error)
            elif kind == "end" and self.status != "ended":
                if self.disturbance and self.disturbance.phase in ("queued", "active", "recovery"):
                    self.disturbance.phase = "interrupted"
                    self._event("disturbance_interrupted", **self.disturbance.summary(self.seq))
                self._cancel_update()
                self.generation += 1
                self.status = "ended"
                self.lease_paused = False
            else:
                raise ValueError(f"当前状态 {self.status} 无法执行 {kind}")
            self._event(kind if kind != "reference" else "reference_queued", key=key, value=value, **details)
            answer = dict(key=key, run=self.run, status=self.status, seq=self.seq)
            self.seen[key] = answer
            self.lock.notify_all()
            return answer

    def prepare_update(self, *, automatic=False):
        with self.lock:
            if self.status == "ended" or self.closed:
                raise ValueError("会话已结束")
            if self.update and not self.update["cancel"].is_set():
                return self.update, None
            if self.trial_previous is not None or self.pending is not None:
                raise ValueError("请等待候选模型的首个控制区间完成")
            rows = list(self.history)
            if len(rows) < UPDATE_BATCH:
                raise ValueError(f"在线更新至少需要 {UPDATE_BATCH} 个已完成且可信的连续区间，当前 {len(rows)}")
            x, y = zip(*rows)
            trajectory = Trajectories(np.array([x]), np.array([y]))
            info = dict(id=None, run=self.run, base=self.model.version, cutoff=self.seq, automatic=automatic,
                        phase="queued", first=self.seq-len(rows)+1, cancel=Event())
            self.update_cutoff = self.seq
            self.update = info
            return info, (self.model, trajectory)

    def finish_update(self, info, candidate, validation):
        with self.lock:
            if self.closed or self.update is not info or self.run != info["run"] or self.status == "ended":
                return False
            if info["cancel"].is_set() or candidate is None or self.model.version != info["base"]:
                detail = validation.get("reason") or " · ".join(validation.get("reasons", []))
                failed = "reason" in validation
                self.last_online_result = dict(state="failed" if failed else "rejected", seq=self.seq, detail=detail)
                self.pending = self.update = None
                self.lock.notify_all()
                self._event("update_failed" if failed else "update_rejected", job=info["id"], report=validation)
                return False
            self._event("validation_passed", job=info["id"], base=info["base"],
                        version=candidate.version, active_interval=self.seq+1 if self.in_flight else None, report=validation)
            info["phase"] = "awaiting_cycle"
            self.pending = candidate, info, self.event_number
            return True

    def export(self):
        with self.lock:
            run, log, seq, event = self.run, self.log, self.seq, self.event_number
            config = asdict(self.settings)
            barrier = log.barrier()
        records = log.records(barrier, seq, event)
        intervals = [item for item in records if item["kind"] == "sample" and item["seq"]]
        failures = [item for item in records if item.get("event") == "failed"]
        missed = sum(item.get("deadline_missed", False) for item in failures)
        solve_ms = [row["solve_ms"] for row in intervals]
        compute_ms = [row["cycle_ms"] for row in intervals]
        wall_ms = sum(row["wall_ms"] for row in intervals)
        attempts = len(intervals)+len(failures)
        timing = dict(cycles=len(intervals), failed=len(failures), deadline_misses=missed,
                      overrun_rate=missed/attempts if attempts else 0,
                      wall_budget_ms=60000*SAMPLE_MIN/config["speed"],
                      achieved_speed=len(intervals)*60000*SAMPLE_MIN/wall_ms if wall_ms else None,
                      solve_p95_ms=float(np.percentile(solve_ms, 95)) if solve_ms else None,
                      compute_p95_ms=float(np.percentile(compute_ms, 95)) if compute_ms else None)
        return dict(run=run, config=config, seq=seq, event_id=event, timing=timing, records=records)

    def _select_cycle(self):
        if self.pending:
            candidate, info, accepted = self.pending
            if not info["cancel"].is_set() and info["run"] == self.run and self.model.version == info["base"]:
                self.trial_previous = self.model
                self.model, self.warm = candidate, None
                if self.update is info:
                    self.last_online_result = dict(state="activated", seq=self.seq+1, version=candidate.version)
                self._event("activated", job=info["id"], old=self.trial_previous.version, new=candidate.version,
                            ready_event=accepted, first_interval=self.seq+1)
            else:
                self._event("update_discarded", job=info["id"])
            self.pending = self.update = None
        if self.pending_ref:
            key, self.reference = self.pending_ref
            self._event("reference_applied", key=key, value=self.reference, first_interval=self.seq+1)
            self.pending_ref = None
        return self.trial_previous

    def _run(self):
        while True:
            try:
                self._loop()
                return
            except Exception as exc:
                with self.lock:
                    if self.closed:
                        return
                    self.in_flight = False
                    self.status, self.error = "paused", str(exc)
                    self.lock.wait_for(lambda: self.closed or self.status in ("ready", "running"))
                    if self.closed:
                        return

    def _loop(self):
        step(NOMINAL, NOMINAL_FLOW)  # Warm the numerical kernel before the first timed interval.
        while True:
            with self.lock:
                if self.closed:
                    return
                idle = time.monotonic()-self.last_seen
                if self.leased and not self.owner_connections and idle > 120:
                    self._cancel_update()
                    self._event("lease_expired")
                    self.status, self.closed = "ended", True
                    self.lock.notify_all()
                    self.log.close()
                    return
                if self.leased and not self.owner_connections and idle > 20 and self.status == "running":
                    self.lease_paused = True
                    self.status = "paused"
                    self._event("lease_paused")
                if self.status != "running":
                    self.lock.wait(1)
                    continue
                old = self._select_cycle()
                trial = self.disturbance
                inlet = INLET
                if trial is not None:
                    event = trial.boundary(self.seq, self.state, self.previous, self.reference)
                    if event:
                        self._event(event, first_interval=self.seq+1, **trial.summary(self.seq))
                    if trial.phase == "active":
                        inlet = trial.input
                    if not trial.observing:
                        trial = None
                self.in_flight = True
                model = self.model
                generation = self.generation
                x, previous, ref, warm, settings = self.state.copy(), self.previous, self.reference, self.warm, self.settings
            start = time.perf_counter()
            deadline = start+60*SAMPLE_MIN/settings.speed
            attempts = [model.version]
            failure = None
            rollback = False
            try:
                try:
                    seq, diag = solve(model, x, previous, ref, settings, warm, inlet=inlet, deadline=deadline)
                except (RuntimeError, ValueError, TimeoutError) as exc:
                    if old is None:
                        raise
                    rollback = True
                    failure = str(exc)
                    if time.perf_counter() >= deadline:
                        raise
                    model = old
                    attempts.append(model.version)
                    seq, diag = solve(model, x, previous, ref, settings, None, inlet=inlet, deadline=deadline)
                cmd = float(seq[0])
                applied = apply_flow(cmd)
                nxt = step(x, applied, inlet)
                comparison = trial.compare(inlet) if trial is not None else None
                compute_ms = (time.perf_counter()-start)*1000
                if time.perf_counter() > deadline:
                    raise TimeoutError("总周期计算超时")
                error = None
            except Exception as exc:
                error = str(exc)
            deadline_missed = time.perf_counter() > deadline
            with self.lock:
                if self.closed or self.generation != generation or self.status != "running":
                    continue
                if old is not None and error and not rollback:
                    rollback = True
                    failure = error
                if rollback:
                    if (self.last_online_result and self.last_online_result["state"] == "activated"
                            and self.last_online_result["seq"] == self.seq+1):
                        self.last_online_result = dict(state="rollback", seq=self.seq+1, version=old.version,
                                                       detail=failure)
                    self.model, self.warm, self.trial_previous = old, None, None
                    self._event("rollback", reason=failure, attempts=attempts, restored=old.version)
                if error:
                    self.in_flight = False
                    self.status, self.error = "paused", error
                    self._event("failed", reason=error, attempts=attempts, deadline_missed=deadline_missed)
                    continue
                self.lock.wait_for(lambda: self.closed or self.generation != generation,
                                   timeout=max(0.0, deadline-time.perf_counter()))
                if self.closed or self.generation != generation or self.status != "running":
                    continue
                self.in_flight = False
                next_number = self.seq + 1
                diagnostics = {**diag, "attempts": attempts, "wall_ms": 1000*(time.perf_counter()-start)}
                try:
                    if trial is not None:
                        diagnostics.update(trial.sample_fields(comparison))
                    self._sample(cmd, applied, model.version, diagnostics, compute_ms, nxt, next_number, inlet)
                except (OSError, RuntimeError) as exc:
                    self.status, self.error = "paused", str(exc)
                    continue
                self.trial_previous = None
                self.state, self.previous, self.warm = nxt, applied, seq
                self.seq = next_number
                self.history.append((np.r_[x, applied, inlet], nxt.copy()))
                if trial is not None and trial.commit(next_number, nxt[1], ref, comparison):
                    self._event("disturbance_completed", **trial.summary(next_number))
                if self._update_due():
                    self.lock.notify_all()

    def close(self):
        with self.lock:
            if not self.closed:
                self._cancel_update()
                self.status, self.closed = "ended", True
                self.lock.notify_all()
        self.worker.join()
        if self.update_worker is not None:
            self.update_worker.join()
        self.log.close()
