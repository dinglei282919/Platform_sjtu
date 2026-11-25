"""Paired, independently integrated CSTR trials using a fixed offline QA casebook."""
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import cstr
from .dnn_mpc import model as predictor
from .fdi_control import (DT, FLOW_NOISE, JACKET_NOISE, Q_MAX, Q_REF, TEMP_NOISE, T_REF,
                          ExecutionLoop, Gate, LoopSettings, MeasurementLoop, TemperaturePI, plant_interval)


BOOK = json.loads((Path(__file__).resolve().parents[2] / "assets/cstr/fdi-samples.json").read_text("utf-8"))
DEMO_VIEWS = json.loads((Path(__file__).resolve().parents[2] / "assets/cstr/fdi-demo.json").read_text("utf-8"))
SCENARIOS = {
    "measurement": ("readout", "healthy", "inlet", "readout_soft", "noisy_event", "mismatch_clean",
                    "mismatch_event", "missing_jacket"),
    "execution": ("delivery", "healthy", "inlet", "delivery_soft", "heat_only", "mixed", "noisy_event",
                  "mismatch_clean", "mismatch_event", "headroom", "missing_jacket"),
}
GROUPS = {
    "measurement": {"raw": "普通 PI", "guard": "一致性校验 + PI", "fixed": "固定权重 + PI"},
    "execution": {"pi": "普通 PI", "full": "两层完整补偿", "bare": "两项关闭",
                  "outer": "仅釜侧补偿", "inner": "仅夹套侧补偿"},
}
MODEL_VERSION = "d68f6dcc92351e821f9f"
RECOVERY_STEPS = 80
FIXED_PRIMARY_WEIGHT = 0.0


@dataclass(frozen=True)
class Config:
    side: str = "measurement"
    scenario: str = "readout"
    seed: int = 1301
    speed: float = 60.

    @classmethod
    def parse(cls, values=None):
        fields = dict(values or {})
        side = fields.get("side", "measurement")
        if not isinstance(side, str) or side not in GROUPS:
            raise ValueError("未知试验端别")
        fields.setdefault("scenario", SCENARIOS[side][0])
        cfg = cls(**fields)
        if cfg.scenario not in SCENARIOS[side]:
            raise ValueError("所选端别没有这个样本")
        if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**31:
            raise ValueError("种子须为 0 至 2147483647 的整数")
        if type(cfg.speed) not in (float, int) or not 1 <= cfg.speed <= 1200:
            raise ValueError("推进倍率须在 1 至 1200 之间")
        return cfg


def options():
    return dict(scenarios={side: {k: BOOK["cases"][k]["label"] for k in cases}
                           for side, cases in SCENARIOS.items()}, groups=GROUPS,
                sample_min=DT, segment_samples=BOOK["length"], recovery_samples=RECOVERY_STEPS,
                defaults={side: asdict(Config.parse({"side": side})) for side in GROUPS})


def signal_link(side, scenario):
    channel = "readout_K" if side == "measurement" else "delivery_L_min"
    return side if channel in BOOK["cases"][scenario] else None


class Branch:
    def __init__(self, initial, controller):
        self.x = initial.copy()
        self.controller = controller
        self.last_actual = Q_REF
        self.last_inlet = None
        self.last = None
        self.failure = None

    def sample(self, k, name, noise, inlet, instrument_offset, flow_offset, load, case, terminal):
        c, temp, jacket = map(float, self.x)
        tm, tcm = temp + noise[0] + instrument_offset, jacket + noise[1]
        row = dict(seq=k, C=c, T=temp, Tc=jacket, Tm=tm, Tcm=float(tcm) if np.isfinite(tcm) else None,
                   Ci=inlet[0], Ti=inlet[1], Tci=inlet[2], T_ref=T_REF,
                   Qc=None, Qc_cmd=None, previous_Qc=self.last_actual, load=load, source="主釜温", valid=True)
        loop = self.controller
        if isinstance(loop, MeasurementLoop):
            read = self.last_actual + noise[2]
            fb, witness, residual, weight = loop.observe(tm, tcm, read, self.last_inlet)
            row.update(feedback=fb, witness=witness if np.isfinite(witness) else None,
                       residual=residual if np.isfinite(residual) else None, weight=weight,
                       active=loop.active, flow_read_previous=read, sigma_T=float(np.sqrt(loop.cov[1, 1])),
                       source="主釜温" if weight == 1 else "夹套估计" if weight == 0 else "加权融合")
            command = None if terminal else loop.pi.command(fb)
        elif name == "pi":
            row["feedback"] = tm
            command = None if terminal else loop.command(tm)
        else:
            row["source"] = "釜温与夹套测温"
            command = None
            if not terminal:
                command, diagnostics = loop.command(tm, tcm, inlet)
                row.update(diagnostics)
        if command is not None:
            capacity = Q_REF + case["headroom_L_min"] if "headroom_L_min" in case else Q_MAX
            actual = cstr.apply_flow(min(command, capacity) + flow_offset)
            next_x = plant_interval(self.x, actual, inlet, case.get("exchange", 1.), load, cstr.INTEGRATION_MIN)
            if not np.isfinite(next_x).all() or min(next_x) <= 0:
                raise FloatingPointError("对象积分越出物理域")
            row.update(Qc=actual, Qc_cmd=command, capacity_hit=command > capacity,
                       command_limited=command in cstr.FLOW_LIMITS)
            self.x, self.last_actual, self.last_inlet = next_x, actual, inlet
        return row


class Comparison:
    """Row k measures x[k] and records the decision for [k,k+1). Labels stay here."""

    def __init__(self, config, *, continuous=False):
        self.config = config
        self.continuous = continuous
        self._select_sample(config.scenario)
        self.rng = np.random.default_rng(config.seed)
        self.initial = cstr.NOMINAL + self.rng.normal(size=3)*[0.0007, 0.07, 0.04]
        self.model_version = None
        model = None
        if config.side == "execution":
            model = predictor.load()
            if model.version != MODEL_VERSION:
                raise ValueError("执行端需要通过本试验验收的冻结 DNN 权重")
            self.model_version = model.version
        self.branches = {}
        for name in GROUPS[config.side]:
            if config.side == "measurement":
                loop = MeasurementLoop(name, Gate(), FIXED_PRIMARY_WEIGHT)
            elif name == "pi":
                loop = TemperaturePI()
            else:
                loop = ExecutionLoop(model, name, LoopSettings())
            self.branches[name] = Branch(self.initial, loop)
        self.seq = -1
        self.segment = 0
        self.start = self.stop = None
        self.healthy_twin = None
        self.response_seq = None

    def _select_sample(self, scenario):
        self.scenario = scenario
        self.case = BOOK["cases"][scenario]
        points = np.array(BOOK["profiles"][self.case.get("profile", "flat")])
        self.profile = np.interp(np.arange(BOOK["length"]), points[:, 0], points[:, 1])

    def advance(self, action=None, *, scenario=None):
        k = self.seq + 1
        if action == "begin":
            if scenario is not None and scenario != self.scenario:
                self._select_sample(scenario)
            self.segment += 1
            self.start, self.stop = k, k + len(self.profile)
            self.response_seq = None
            if self.config.side == "execution":
                # Evaluate this exposure against the same pre-exposure plant and controller state.
                full = self.branches["full"]
                model = full.controller.model
                self.healthy_twin = deepcopy(full, {id(model): model})
        elif action == "finish":
            self.stop = k
        active = self.start is not None and self.start <= k < self.stop
        evaluation_end = self.stop is not None and k == self.stop + RECOVERY_STEPS
        terminal = evaluation_end and not self.continuous
        if self.stop is None or k < self.stop:
            phase = "active" if active else "healthy"
        else:
            phase = "recovery" if k <= self.stop + RECOVERY_STEPS else "monitoring"
        offset = k-self.start if active else None
        scale = self.profile[offset] if active else 0.
        case, side = self.case, self.config.side
        noise = self.rng.normal(size=3) * [TEMP_NOISE, JACKET_NOISE, FLOW_NOISE]
        noise[:2] *= case.get("noise_scale", 1.)
        if active and offset == case.get("missing_jacket_at"):
            noise[1] = np.nan
        inlet = cstr.INLET.copy()
        inlet[1] += scale * case.get("feed_K", 0.)
        readout = scale * case.get("readout_K", 0.) if side == "measurement" else 0.
        delivery = scale * case.get("delivery_L_min", 0.) if side == "execution" else 0.
        heat = scale * case.get("heat_K_min", 0.)
        groups, events = {}, []
        if k == self.start or k == self.stop:
            events.append(dict(event="segment_start" if active else "segment_end",
                               segment=self.segment, scenario=self.scenario))
        for name, branch in self.branches.items():
            if branch.failure is None:
                try:
                    current = branch.sample(k, name, noise, inlet, readout, delivery, heat, case, terminal)
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    branch.failure = dict(seq=k, reason=str(exc))
                    events.append(dict(event="feedback_unavailable", group=name, reason=str(exc)))
                else:
                    last = branch.last
                    if name == "guard" and last and current["source"] != last["source"]:
                        events.append(dict(event="feedback_source", group=name, source=current["source"]))
                    branch.last = current
            if branch.failure:
                groups[name] = {**(branch.last or {}), "valid": False, "failure": branch.failure}
            else:
                groups[name] = branch.last
        if self.healthy_twin is not None and self.healthy_twin.failure is None and not evaluation_end and phase != "monitoring":
            # Use identical instruments, known inlet and capacity. Omit only the
            # prescribed delivery/thermal record when evaluating its effect.
            try:
                twin = self.healthy_twin.sample(k, "full", noise, inlet, 0., 0., 0., case, False)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                self.healthy_twin.failure = dict(seq=k, reason=str(exc))
            else:
                full = groups["full"]
                if full["valid"] and self.start is not None and self.response_seq is None:
                    changed = any(abs(full[key]-twin[key]) > 1e-9 for key in ("inner_delta_q", "load_used"))
                    if changed:
                        self.response_seq = k
                        events.append(dict(event="compensation_response", group="full"))
        self.seq = k
        return dict(seq=k, t=round(k*DT, 10), phase=phase, segment=self.segment, scenario=self.scenario,
                    link=signal_link(side, self.scenario),
                    groups=groups, evaluation_end=evaluation_end, terminal=terminal), events
