"""Four synchronized PI loops on the common CSTR, with frozen normal-data maps."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from numba import njit

from .cstr import (FLOW_LIMITS, INLET, INTEGRATION_MIN, NOMINAL, NOMINAL_FLOW, SAMPLE_MIN,
                   STATE_NAMES, integrate)

GROUPS = {"normal": "正常 PI", "fault": "退化 PI", "scsc": "退化 PI + SCSC", "ccsc": "退化 PI + CCSC"}
SCENARIOS = {"catalyst": "轻微催化失活", "htc": "夹套换热衰减"}
MODES = {"event": "逐采样事件触发", "batch": "10 min 批均值"}
SIGNALS = {"C": ["釜内浓度", "mol/L"], "T": ["釜内温度", "K"], "Tc": ["夹套出口温度", "K"],
           "Ci": ["进料浓度", "mol/L"], "Ti": ["进料温度", "K"], "Tci": ["冷却液入口温度", "K"],
           "Qc": ["冷却液流量", "L/min"]}
LOWER = np.array([0.2, 280., 250.])
UPPER = np.array([2., 420., 410.])
LIMIT = 0.005 * INLET
WIDTHS = tuple(map(float, 0.005 * NOMINAL))
REFERENCES = tuple(map(float, NOMINAL))
# Local tuning ranges for the healthy operating point, in each measured channel's unit.
TRIGGER_LIMITS = {
    "C": {"reference": (0.099, 0.101), "half_width": (0.0001, 0.002)},
    "T": {"reference": (430.7, 431.1), "half_width": (0.1, 3.0)},
    "Tc": {"reference": (416.5, 417.0), "half_width": (0.1, 3.0)},
}
FAULT_MIN = 200.
END_MIN = 1200.
BATCH_STEPS = 100
END_STEP = round(END_MIN / SAMPLE_MIN)
POST_FAULT_MIN = END_MIN - FAULT_MIN
TOLERANCE = np.array([0.001, 0.5])
FAULT_PROFILE = {"catalyst": ("a", 0.0005, 0.95), "htc": ("b", 0.001, 0.)}
EFFECT_OUTPUT = {"catalyst": "C", "htc": "T"}
PI_KP, PI_KI = 1., 5.

# The bounds are the ones used by the platform's numeric editors.
def _parameter(lo, hi, step, decimals, unit, help_text):
    return dict(min=lo, max=hi, step=step, decimals=decimals,
                hint=f"可调范围 {lo:,} 至 {hi:,}{unit}。{help_text}")


COMPENSATION_PARAMETERS = {
    "interval": _parameter(SAMPLE_MIN, 10**12, .1, 1, " min", "逐采样事件触发使用，须为 0.1 min 的整数倍。"),
    "gain": _parameter(-10**9, 10**9, .0005, 8, "", "有符号的入口修正倍率。0 表示不修正入口，实际改变量受限幅约束。"),
    "seed": _parameter(0, 2**31-1, 1, 0, "（整数）", "相同种子可复现正常数据与拟合映射。"),
    "speed": _parameter(1, 1200, 10, 1, "×", "每周期推进 0.1 仿真分钟。"),
}


@dataclass(frozen=True)
class Config:
    scenario: str = "catalyst"
    mode: str = "event"
    widths: tuple = WIDTHS
    scsc_interval: float = 2.
    ccsc_interval: float = 2.
    scsc_gain: float = -0.3
    ccsc_gain: float = -0.3
    speed: float = 600.
    seed: int = 42
    references: tuple = REFERENCES

    @classmethod
    def parse(cls, options=None):
        args = dict(options or {})
        scenario, mode = args.get("scenario", "catalyst"), args.get("mode", "event")
        if scenario not in SCENARIOS or mode not in MODES:
            raise ValueError("请选择有效的退化场景和补偿调度")
        if scenario == "htc":
            args.setdefault("ccsc_interval", 5.)
            args.setdefault("ccsc_gain", -0.001 if mode == "event" else -0.3)
        if set(args) - set(cls.__dataclass_fields__):
            raise ValueError("存在未知自愈参数")
        def finite(value):
            return type(value) is int or (type(value) is float and math.isfinite(value))
        for key, kind, default, label in (("references", "reference", REFERENCES, "参考点"),
                                          ("widths", "half_width", WIDTHS, "半宽")):
            values = args.get(key, default)
            if not isinstance(values, (tuple, list)) or len(values) != len(STATE_NAMES):
                raise ValueError(f"请依次提供 C、T、Tc 的{label}")
            for channel, value in zip(STATE_NAMES, values):
                lo, hi = TRIGGER_LIMITS[channel][kind]
                if not finite(value) or not lo <= value <= hi:
                    raise ValueError(f"{channel} {label}须在 {lo:g} 至 {hi:g} {SIGNALS[channel][1]} 范围内")
            args[key] = tuple(values)
        cfg = cls(**args)
        for v in (cfg.scsc_interval, cfg.ccsc_interval):
            if not finite(v) or not SAMPLE_MIN <= v <= COMPENSATION_PARAMETERS["interval"]["max"]:
                raise ValueError(COMPENSATION_PARAMETERS["interval"]["hint"])
            fraction = v % 1.
            if not math.isclose(fraction, round(fraction/SAMPLE_MIN)*SAMPLE_MIN, abs_tol=1e-10, rel_tol=0):
                raise ValueError("最小补偿间隔须为 0.1 min 的整数倍")
        gain = COMPENSATION_PARAMETERS["gain"]
        if not all(finite(v) and gain["min"] <= v <= gain["max"] for v in (cfg.scsc_gain, cfg.ccsc_gain)):
            raise ValueError(gain["hint"])
        if not finite(cfg.speed) or not 1 <= cfg.speed <= 1200:
            raise ValueError("推进倍率须在 1–1200 之间")
        if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**31:
            raise ValueError("拟合种子须为 0–2147483647 的整数")
        return cfg


def options():
    # Older web clients read a single pair of numeric bounds.
    ranges = [bounds for channel in TRIGGER_LIMITS.values() for bounds in channel.values()]
    return dict(groups=GROUPS, scenarios=SCENARIOS, modes=MODES, signals=SIGNALS, nominal=NOMINAL.tolist(),
                flow_limits=FLOW_LIMITS, inlet_limits=[LOWER.tolist(), UPPER.tolist()], period=SAMPLE_MIN,
                fault_mode="manual", post_fault_min=POST_FAULT_MIN, defaults=asdict(Config()),
                parameters=COMPENSATION_PARAMETERS, trigger_limits=TRIGGER_LIMITS,
                band_limits=(min(lo for lo, _ in ranges), max(hi for _, hi in ranges)),
                presets={s: {m: asdict(Config.parse(dict(scenario=s, mode=m))) for m in MODES} for s in SCENARIOS})


@njit(cache=True)
def pi_step(state, integral, inlet, activity, heat):
    error = state[1] - NOMINAL[1]
    raw = integral + PI_KP * error
    lo, hi = FLOW_LIMITS
    flow = min(max(raw, lo), hi)
    if not ((raw >= hi and error > 0) or (raw <= lo and error < 0)):
        integral += PI_KI * SAMPLE_MIN * error
    return integrate(state, flow, inlet, SAMPLE_MIN, INTEGRATION_MIN, activity, heat), integral, flow


@njit(cache=True)
def normal_data(holds):
    inputs = np.empty((len(holds) * BATCH_STEPS // 2, 3))
    outputs = np.empty_like(inputs)
    state, integral = NOMINAL.copy(), NOMINAL_FLOW
    row = 0
    for inlet in holds:
        for k in range(BATCH_STEPS):
            state, integral, _ = pi_step(state, integral, inlet, 1., 1.)
            if k >= BATCH_STEPS // 2:
                inputs[row], outputs[row] = inlet, state
                row += 1
    return inputs, outputs


def roots(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    root = np.sqrt(values)
    return (vectors * root) @ vectors.T, (vectors * (1. / root)) @ vectors.T


@lru_cache(maxsize=8)
def fit_maps(seed):
    holds = INLET * (1 + np.random.default_rng(seed).uniform(-0.005, 0.005, (100, 3)))
    x, y = normal_data(holds)
    dx, dy = x.std(axis=0, ddof=1), y.std(axis=0, ddof=1)
    nx, ny = (x-x.mean(axis=0))/dx, (y-y.mean(axis=0))/dy
    n = len(x)-1
    ridge = 1e-8 * np.eye(3)
    xx, yy, xy = nx.T @ nx/n+ridge, ny.T @ ny/n+ridge, nx.T @ ny/n
    xr, xi = roots(xx)
    _, yi = roots(yy)
    u, singular, vt = np.linalg.svd(xi @ xy @ yi)
    singular = np.clip(singular, 0., 1.)
    pearson = np.clip(xy, -1., 1.)
    mapping = yi @ vt.T @ np.diag(singular) @ u.T @ xr
    diagnostic = dict(seed=seed, samples=len(x), input_rank=int(np.linalg.matrix_rank(nx)),
                      output_rank=int(np.linalg.matrix_rank(ny)), input_std=dx.tolist(), output_std=dy.tolist(),
                      canonical_correlations=singular.tolist(), pearson_input_output=pearson.tolist(),
                      standardized_mapping=mapping.tolist(), standardized_output_covariance_condition=float(np.linalg.cond(yy)),
                      weights=[1., 1., 0.05], ridge=1e-8, centering="reference", residual="R1=0")
    digest = hashlib.sha256(json.dumps(diagnostic, sort_keys=True).encode()).hexdigest()[:16]
    diagnostic["version"] = "pi-correlation-v1-" + digest
    return pearson, mapping, dx, dy, diagnostic


class Comparison:
    """Prepare one interval from completed measurements. Commit only after it has been recorded."""
    def __init__(self, config, *, fault_seq=round(FAULT_MIN / SAMPLE_MIN)):
        self.config = config
        # Offline benchmarks use the published onset; a live run starts unarmed.
        self.fault_seq = fault_seq
        self.seq = 0
        self.state = np.tile(NOMINAL, (4, 1))
        self.integral = np.full(4, NOMINAL_FLOW)
        self.inlets = np.tile(INLET, (4, 1))
        self.batch = np.zeros((BATCH_STEPS, 4, 3))
        self.last_update = [None]*4
        self.references = np.array(config.references)
        self.widths = np.array(config.widths)
        self.lower_band, self.upper_band = self.references-self.widths, self.references+self.widths
        self.gaps = [0, 0, *(round(v/SAMPLE_MIN) for v in (config.scsc_interval, config.ccsc_interval))]
        self.gains = [0., 0., config.scsc_gain, config.ccsc_gain]
        self.maps = None

    def initial(self):
        return self._sample(0, self.state, self.inlets, self.integral, 1., 1., [])

    def _sample(self, seq, states, inlets, flows, activity, heat, events):
        groups = {}
        for i, key in enumerate(GROUPS):
            c, t, tc = states[i]
            ci, ti, tci = inlets[i]
            groups[key] = dict(C=c, T=t, Tc=tc, Ci=ci, Ti=ti, Tci=tci, Qc=flows[i], T_ref=NOMINAL[1],
                               a=activity if i else 1., b=heat if i else 1.,
                               outside=[name for name, hit in zip(STATE_NAMES, (states[i]<self.lower_band) | (states[i]>self.upper_band)) if hit])
        return dict(seq=seq, t=seq*SAMPLE_MIN, start_min=max(0, seq-1)*SAMPLE_MIN, groups=groups, compensation=events)

    def prepare(self):
        if self.maps is None:
            self.maps = fit_maps(self.config.seed)
        k = self.seq
        at = k * SAMPLE_MIN
        age = max(0., at - self.fault_seq*SAMPLE_MIN) if self.fault_seq is not None else 0.
        factor, rate, floor = FAULT_PROFILE[self.config.scenario]
        degraded = max(floor, np.exp(-rate*age))
        activity, heat = (degraded, 1.) if factor == "a" else (1., degraded)
        inlets, last = self.inlets.copy(), self.last_update.copy()
        batch = self.config.mode != "event"
        boundary = k > 0 and k % BATCH_STEPS == 0
        events = []
        if k and (not batch or boundary):
            pearson, cca, dx, dy, _ = self.maps
            for i, name in ((2, "scsc"), (3, "ccsc")):
                measurements = self.batch[:, i] if batch else self.state[i:i+1]
                observed = measurements.mean(axis=0)
                err = observed - self.references
                hit = (observed < self.lower_band) | (observed > self.upper_band)
                if not hit.any() or (not batch and last[i] is not None and k-last[i] < self.gaps[i]):
                    continue
                raw = (((err/self.references * (1.,1.,0.05)) @ pearson.T) * inlets[i] if i == 2
                       else ((((measurements-self.references)/dy) @ cca)*dx).mean(axis=0))
                requested = self.gains[i]*raw
                if not np.isfinite(requested).all():
                    raise FloatingPointError("补偿计算超出有限数值范围")
                limited = np.clip(requested, -LIMIT, LIMIT)
                before = inlets[i].copy()
                inlets[i] = np.clip(before+limited, LOWER, UPPER)
                events.append(dict(group=name, at=at, seq=k, first_seq=k+1, first_sample_min=(k+1)*SAMPLE_MIN,
                                   channels=[ch for ch, flag in zip(STATE_NAMES, hit) if flag], error=err.tolist(),
                                   requested=requested.tolist(), applied=(inlets[i]-before).tolist(), inlets=inlets[i].tolist(),
                                   step_limited=(abs(requested)>LIMIT).tolist(),
                                   inlet_limited=((before+limited<LOWER)|(before+limited>UPPER)).tolist()))
                last[i] = k
        state, integral, flow = np.empty_like(self.state), np.empty(4), np.empty(4)
        for i in range(4):
            state[i], integral[i], flow[i] = pi_step(self.state[i], self.integral[i], inlets[i],
                                                      activity if i else 1., heat if i else 1.)
        if not np.isfinite(state).all() or np.min(state)<=0:
            raise FloatingPointError("CSTR 积分超出有限正值域")
        return self._sample(k+1, state, inlets, flow, activity, heat, events), (state, integral, inlets, last)

    def commit(self, next_state):
        self.state, self.integral, self.inlets, self.last_update = next_state
        self.batch[self.seq % BATCH_STEPS] = self.state
        self.seq += 1
