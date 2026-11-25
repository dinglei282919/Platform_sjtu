"""Independent PI and heat-observer trials on the shared three-state CSTR."""
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import cstr
from .thermal import plant_interval

DT = cstr.SAMPLE_MIN
T_REF, TC_REF = map(float, cstr.NOMINAL[1:])
Q_REF = cstr.NOMINAL_FLOW
START, END = 50, 90
ARMS = {"raw": "按收到值更新", "guard": "校验采样时刻", "candidate": "门限重估",
        "continuous": "逐拍重估", "fresh": "全程当前测温", "pi": "仅本地 PI"}
LOCAL_THRESHOLD_K = 0.05
LOCAL_REQUIRED_SAMPLES = 2
IMMEDIATE = "immediate"
_records = json.loads((Path(__file__).resolve().parents[2] / "assets/cstr/replay-observations.json").read_text("utf-8"))
FIXTURE = {k: (source, t, tc) for k, source, t, tc in _records}
RECOVERY_STEPS = END - START - len(FIXTURE)


@dataclass(frozen=True)
class Case:
    label: str
    delivery: str = "record"
    length: int = 10
    heat: float = 0.6
    removal: int | None = None
    noise: bool = False
    exchange: float = 1.0


CASES = {
    "constant": Case("恒定热负荷"),
    "removed": Case("旧样本段内撤除热负荷", removal=53),
    "healthy": Case("健康通信", delivery="current"),
    "steady": Case("稳态零热负荷", heat=0.0),
    "delay": Case("正常通信延迟一拍", delivery="delay"),
    "noise": Case("带噪测温", noise=True),
    "mismatch": Case("换热系数偏低", exchange=0.99),
    "short_gap": Case("连续缺测五拍", delivery="missing", length=5),
    "long_gap": Case("连续缺测二十拍", delivery="missing", length=20),
    "unknown": Case("源采样号缺失", delivery="unknown"),
    "local_missing": Case("本地当前测温缺失", delivery="local_missing"),
}
DEMO_CASES = {k: CASES[k].label for k in ("constant", "removed")}


@lru_cache(maxsize=1)
def heat_sensitivity():
    def temperature(load):
        return plant_interval(cstr.NOMINAL, Q_REF, cstr.INLET, 1., load, cstr.INTEGRATION_MIN)[1]
    return float((temperature(.05) - temperature(-.05)) / .1)


class HeatObserver:
    def __init__(self, policy, substep):
        self.policy, self.substep = policy, substep
        self.state = cstr.NOMINAL.copy()
        self.heat, self.weight, self.missed = 0., 1., 0

    def correct_heat(self, innovation):
        self.heat = float(np.clip(self.heat + .25 * innovation / heat_sensitivity(), -1.5, 1.5))

    def update(self, k, observation, previous_flow):
        source, t, tc = observation
        if k:
            self.state = plant_interval(self.state, previous_flow, cstr.INLET, 1., self.heat, self.substep)
        prior = float(self.state[1])
        innovation = None
        usable = t is not None and tc is not None and (self.policy == "raw" or source == k)
        if usable:
            if k and not self.missed:
                innovation = t - prior
                self.correct_heat(innovation)
            action = "reanchor" if self.missed else ("current" if source == k else "as_received")
            self.state[1:] = t, tc
            self.missed = 0
            self.weight = min(1., round(self.weight + .2, 10))
        else:
            self.missed += 1
            if self.policy == IMMEDIATE:
                self.weight = 0.
            elif self.policy != "raw" and self.missed > 10:
                self.weight = max(0., round(self.weight - .2, 10))
            action = "hold" if self.weight == 1. else "fade"
        return prior, innovation, action


class LocalHeatObserver(HeatObserver):
    """Use the PI's current vessel thermometer while the remote copy is unusable."""

    def __init__(self, policy, substep):
        super().__init__(policy, substep)
        self.local_state = None
        self.streak = 0
        self.updating = False

    def update_with_local(self, k, observation, previous_flow, local_vessel):
        prior, innovation, action = self.update(k, observation, previous_flow)
        prediction = residual = None
        triggered = False
        if not self.missed:
            self.local_state = None
            self.streak = 0
            self.updating = False
        else:
            # The first gap prediction comes from the last trusted observer state.
            # After activation its state also carries each local temperature correction.
            if self.local_state is None or self.updating:
                self.local_state = self.state.copy()
            else:
                self.local_state = plant_interval(self.local_state, previous_flow, cstr.INLET,
                                                  1., self.heat, self.substep)
            prediction = float(self.local_state[1])
            residual = local_vessel - prediction
            sign = int(residual > LOCAL_THRESHOLD_K) - int(residual < -LOCAL_THRESHOLD_K)
            self.streak = self.streak + sign if self.streak * sign > 0 else sign
            triggered = (self.policy == "candidate" and not self.updating
                         and abs(self.streak) >= LOCAL_REQUIRED_SAMPLES)
            self.local_state[1] = local_vessel
            self.updating = self.updating or triggered or self.policy == "continuous"
            if self.updating:
                self.correct_heat(residual)
                self.state = self.local_state.copy()
                action = "local_update"
            prior, innovation = prediction, residual
        diagnostics = dict(local_prediction_K=prediction, local_residual_K=residual,
                           local_streak=self.streak, local_triggered=triggered,
                           local_updating=self.updating)
        return prior, innovation, action, diagnostics


class Branch:
    def __init__(self, name, substep):
        self.name, self.substep = name, substep
        self.state = cstr.NOMINAL.copy()
        self.integral = self.previous_flow = Q_REF
        self.previous_reading = None
        if name in ("candidate", "continuous"):
            self.observer = LocalHeatObserver(name, substep)
        else:
            self.observer = None if name == "pi" else HeatObserver(name, substep)
        self.failure = None

    def sample(self, k, case, noise, heat, scheduled, start=START):
        true_t, true_tc = map(float, self.state[1:])
        local, jacket = true_t + noise[0], true_tc + noise[1]
        if case.delivery == "local_missing" and k == START:
            self.failure = k
        if self.failure is not None:
            return dict(valid=False, failed_at=self.failure, last_valid_seq=self.failure-1, status="local_missing")

        reading = (k, local, jacket)
        delivery = "current"
        if scheduled and self.name != "fresh":
            delivery = case.delivery
            if delivery == "record":
                reading = FIXTURE[START + k - start]
            elif delivery == "delay":
                reading = self.previous_reading
            elif delivery == "missing":
                reading = (None, None, None)
            elif delivery == "unknown":
                reading = (None, local, jacket)
        self.previous_reading = (k, local, jacket)
        source, observed, tc_observed = reading
        estimate = prior = innovation = weight = None
        comp, action = 0., "pi"
        local_diagnostics = {}
        if self.observer is not None:
            if isinstance(self.observer, LocalHeatObserver):
                prior, innovation, action, local_diagnostics = self.observer.update_with_local(
                    k, reading, self.previous_flow, local)
            else:
                prior, innovation, action = self.observer.update(k, reading, self.previous_flow)
            estimate, weight = self.observer.heat, self.observer.weight
            target_tc = TC_REF - weight * estimate / (700000. / 150000.)
            comp = 700. * (T_REF - target_tc) / (target_tc - cstr.INLET[2]) - Q_REF
        error = local - T_REF
        base = self.integral + error
        request = float(base + comp)
        command = cstr.apply_flow(request)
        lower, upper = cstr.FLOW_LIMITS
        if not ((request >= upper and error > 0) or (request <= lower and error < 0)):
            self.integral += 5. * DT * error
        freshness = "missing" if observed is None else "unknown" if source is None else "current" if source == k else "old"
        row = dict(valid=True, k=k, source_k=source, freshness=freshness,
                   C_true_mol_L=float(self.state[0]), T_true_K=true_t, T_local_K=local,
                   Tc_actual_K=true_tc, Tc_local_K=jacket, T_to_observer_K=observed, Tc_to_observer_K=tc_observed,
                   prior_T_K=prior, innovation_K=innovation, heat_true_K_min=heat,
                   heat_est_K_min=estimate, weight=weight, PI_L_min=float(base), comp_L_min=float(comp),
                   request_L_min=request, command_L_min=command, status=action, delivery=delivery,
                   **local_diagnostics)
        self.state = plant_interval(self.state, command, cstr.INLET, case.exchange, heat, self.substep)
        if not np.isfinite(self.state).all() or min(self.state) <= 0:
            raise FloatingPointError("CSTR 状态超出物理域")
        self.previous_flow = command
        return row


class Trial:
    def __init__(self, scenario="constant", seed=1301, *, ablation=False, substep=cstr.INTEGRATION_MIN, manual=False):
        self.scenario = scenario
        self.case = CASES[scenario]
        names = (*ARMS, IMMEDIATE) if ablation else tuple(ARMS)
        self.branches = {name: Branch(name, substep) for name in names}
        self.rng = np.random.default_rng(seed)
        self.seq, self.scheduled = -1, not manual
        self.manual = manual
        self.start = None if manual else START
        self.stop = None if manual else START + self.case.length
        self.remove_heat_at = None if manual else self.case.removal
        self.segment = 0

    def advance(self, action=None, *, scenario=None):
        k = self.seq + 1
        if action == "begin":
            self.scenario = scenario or self.scenario
            self.case = CASES[self.scenario]
            self.start, self.stop = k, k + self.case.length
            self.scheduled = True
            self.remove_heat_at = None if self.case.removal is None else k + self.case.removal - START
        elif action == "finish":
            self.stop = k
            if self.remove_heat_at is not None and k <= self.remove_heat_at:
                self.remove_heat_at = None
        case = self.case
        noise = self.rng.normal(0., (.025, .018)).tolist() if case.noise else (0., 0.)
        heat = case.heat if k >= 30 and (self.remove_heat_at is None or k < self.remove_heat_at) else 0.
        active = self.scheduled and self.start is not None and self.start <= k < self.stop
        if active and k == self.start:
            self.segment += 1
        phase = "healthy"
        if active:
            phase = "active"
        elif self.segment:
            phase = "recovery" if k <= self.stop + RECOVERY_STEPS else "monitoring"
        # The sealed rehearsal also logged the unused observer input of its PI-only arm.
        groups = {name: branch.sample(k, case, noise, heat, active and (not self.manual or name != "pi"), self.start)
                  for name, branch in self.branches.items()}
        self.seq = k
        return dict(seq=k, t=round(k * DT, 10), scheduled=self.scheduled, phase=phase, segment=self.segment,
                    scenario=self.scenario, start_seq=self.start if self.segment else None,
                    stop_seq=self.stop if self.segment else None, groups=groups)


def scores(rows, case, *, start=START, stop=None, end=END):
    """Rates and integrals use completed intervals between the scoring boundaries."""
    result = {}
    if not rows:
        return result
    if stop is None:
        stop = start + case.length
    last = rows[-1]["seq"]
    for name in rows[0]["groups"]:
        points = {r["seq"]: r["groups"][name] for r in rows if r["groups"][name]["valid"]}
        window = [(k, points[k]) for k in range(start, min(last, end)) if k in points]
        observed = [s for k, s in window if k < stop and s["heat_est_K_min"] is not None]
        recovery = 0
        recovered = None
        for k in range(stop, min(last, end) + 1):
            s = points.get(k)
            inside = s and s["heat_est_K_min"] is not None and abs(s["heat_est_K_min"] - s["heat_true_K_min"]) <= .05
            recovery = recovery + 1 if inside else 0
            if recovery == 5:
                recovered = round((k - 4 - stop) * DT, 10)
                break
        benign = [s for _, s in window if name != "pi" and s["delivery"] in ("current", "delay")]
        result[name] = dict(intervals=len(window),
            iae=sum(abs(s["T_true_K"]-T_REF) for _, s in window) * DT if window else None,
            peak=max((abs(s["T_true_K"]-T_REF) for _, s in window), default=None),
            flow_tv=sum(abs(s["command_L_min"]-points[k-1]["command_L_min"]) for k, s in window if k-1 in points) if window else None,
            saturated_min=sum(s["command_L_min"] in cstr.FLOW_LIMITS for _, s in window) * DT if window else None,
            heat_rmse=math.sqrt(sum((s["heat_est_K_min"]-s["heat_true_K_min"])**2 for s in observed)/len(observed)) if observed else None,
            recovery_min=recovered,
            false_hold_rate=sum(s["status"] in ("hold","fade") for s in benign)/len(benign) if benign else None)
    return result
