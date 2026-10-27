"""Incremental FDI reactor regulator; no synthetic disturbance/reference input."""
from collections import deque
from copy import deepcopy
from dataclasses import replace
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from .core import Config, SimulationCancelled, signals, initial_state, switching_mode, SWITCHES
from .kernel import derivatives_compiled
from .attack import parse_attack
from .coordinates import validate_flows, composition_series
from .presentation import build_diagram_series


def _plant_coefficients(mode, r1=.3, r2=.2):
    w1, w2, d1, d2 = (.75, 2., .2, .3) if mode == 1 else (1., 1.8, .4, .3)
    return 1 / w1 + d1, 1 / w2 + d2 - r2 / .5


class ReactorLiveSession:
    """37-state controller, with a moderate finite gain for continuous interaction.

    The offline experiment retains the MATLAB T=1.5, ell=.15 preset. R changes
    explicitly restart the experiment: no fictitious instantaneous composition
    changes are produced by changing the coordinate transform of saved samples.
    """
    def __init__(self, *, step_timeout=1.0):
        if isinstance(step_timeout, bool) or not math.isfinite(step_timeout) or step_timeout <= 0:
            raise ValueError("单步计算时限必须为正数。")
        self.step_timeout = float(step_timeout)
        self.config = Config(prescribed_time=.6, ell=.9, rtol=1e-5, atol=1e-7)
        self.controls = dict(r1=.3, r2=.2, mode=0, enabled=True, attack_enabled=True, attack_scale=1., attack_frequency=.25, attack_phase=0.)
        self.t = 0.
        self.state = initial_state(self.config)
        self.events = []
        self.message = "FDI 闭环组分调节；自动采用 MATLAB 切换表。"
        self._rows = deque(maxlen=600)
        self._warmed = False
        self._record()

    def _effective_config(self):
        return self.config if self.controls["attack_enabled"] else replace(self.config, attack_scale=0.)

    def _record(self):
        mode = self.controls["mode"] or switching_mode(self.t)
        s = signals(self.t, self.state, self._effective_config(), mode)
        row = dict(t=self.t, x1=float(self.state[0]), x2=float(self.state[1]),
                   observer1=float(self.state[2]), observer2=float(self.state[3]),
                   u=float(s["u"]) if self.controls["enabled"] else 0.,
                   mode=float(mode), k=float(s["k"]), gamma=float(s["gamma"]),
                   r1=self.config.r1, r2=self.config.r2,
                   enabled=float(self.controls["enabled"]),
                   attack_enabled=float(self.controls["attack_enabled"]),
                   attack_scale=self._effective_config().attack_scale,
                   attack_frequency=self.config.attack_frequency, attack_phase=self.config.attack_phase)
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError("在线组分或控制输入非有限。")
        if self._rows and self._rows[-1]["t"] == self.t:
            self._rows[-1] = row
        else:
            self._rows.append(row)

    def advance(self, dt=.1, *, cancelled=None):
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("在线推进间隔必须在 (0, 1] 秒内。")
        if cancelled and cancelled():
            raise SimulationCancelled("在线仿真已停止。")
        c = self._effective_config()
        if not self._warmed:
            derivatives_compiled(0., initial_state(c), c.prescribed_time, c.ell, 2, c.r1, c.r2, c.attack_scale, c.attack_frequency, c.attack_phase)
            self._warmed = True
        deadline = time.perf_counter() + self.step_timeout
        end = self.t + float(dt)
        boundaries = [self.t] + sorted({p for p in (*SWITCHES, c.prescribed_time) if self.t < p < end}) + [end]
        candidate = self.state.copy()
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            mode = self.controls["mode"] or switching_mode((start + stop) / 2)
            def rhs(t, state):
                if cancelled and cancelled():
                    raise SimulationCancelled("在线仿真已停止。")
                if time.perf_counter() > deadline:
                    raise RuntimeError("在线积分单步超过时限，保留最近有效状态；请复位。")
                if self.controls["enabled"]:
                    return derivatives_compiled(t, np.ascontiguousarray(state), c.prescribed_time, c.ell, mode, c.r1, c.r2, c.attack_scale, c.attack_frequency, c.attack_phase)
                a, b = _plant_coefficients(mode, c.r1, c.r2)
                dy = np.zeros_like(state)
                dy[0] = state[1] - a * state[0]
                dy[1] = -b * state[1] + c.r1 * (1-c.r2) / (.4*.5) * state[0]
                return dy
            sol = solve_ivp(rhs, (start, stop), candidate, method="BDF", rtol=c.rtol, atol=c.atol, max_step=.02)
            if not sol.success or not np.isfinite(sol.y).all():
                raise RuntimeError("在线积分失败，已保留最近有效状态：" + sol.message)
            candidate = sol.y[:, -1]
        if cancelled and cancelled():
            raise SimulationCancelled("在线仿真已停止。")
        if time.perf_counter() > deadline:
            raise RuntimeError("在线积分单步超过时限，结果未提交。")
        mode = self.controls["mode"] or switching_mode(end)
        if self.controls["enabled"] and not math.isfinite(float(signals(end, candidate, c, mode)["u"])):
            raise FloatingPointError("在线控制输入非有限，结果未提交。")
        self.state, self.t = candidate.copy(), end
        self._record()
        self.message = "FDI 闭环组分调节运行中。" if self.controls["enabled"] else "控制退出：u=0，对象继续响应。"
        return self.snapshot()

    def apply_event(self, kind, value=None):
        if kind not in {"r1", "r2", "mode", "enabled", "reset", "attack", "attack_enabled"}:
            raise ValueError("支持 R1/R2、系统模式、控制投入和复位；FDI 按论文反馈乘性形式注入。")
        if kind == "attack":
            value = parse_attack(value)
        elif kind in {"r1", "r2"}:
            proposed = {"r1": self.config.r1, "r2": self.config.r2, kind: value}
            validate_flows(proposed["r1"], proposed["r2"])
        elif kind == "mode":
            if isinstance(value, bool) or value not in (0, 1, 2):
                raise ValueError("系统模式为 0（MATLAB 自动切换）、1 或 2。")
        elif kind in {"enabled", "attack_enabled"} and not isinstance(value, bool):
            raise ValueError("控制开关必须为布尔值。")
        elif kind == "reset" and value is not None:
            raise ValueError("复位不接收数值。")
        previous = (self.config, self.controls.copy(), self.state.copy(), self.t,
                    deque((r.copy() for r in self._rows), maxlen=600), deepcopy(self.events), self.message)
        try:
            if kind in {"r1", "r2", "reset"}:
                if kind != "reset":
                    self.config = replace(self.config, **{kind: float(value)})
                    self.controls[kind] = float(value)
                self.t = 0.
                self.state = initial_state(self.config)
                self.controls.update(mode=0, enabled=True)
                self._rows.clear()
                self.events.clear()
            elif kind == "attack":
                parameters = {"attack_" + key: val for key, val in value.items()}
                self.config = replace(self.config, **parameters)
                self.controls.update(parameters)
            else:
                self.controls[kind] = value
            self.events.append(dict(time=self.t, kind=kind, value=deepcopy(value)))
            self.events = self.events[-200:]
            self._record()
            self.message = "参数已应用并重新开始实验。" if kind in {"r1", "r2"} else "FDI 参数已在线应用，对象与控制器状态保留。" if kind in {"attack", "attack_enabled"} else "指令已应用。"
            return self.snapshot()
        except Exception:
            self.config, self.controls, self.state, self.t, self._rows, self.events, self.message = previous
            raise

    def snapshot(self):
        data = {key: [row[key] for row in self._rows] for key in self._rows[0]}
        data.update({k: v.tolist() for k, v in composition_series(data).items()})
        data.update(build_diagram_series(data))
        return dict(t=self.t, controls=self.controls.copy(), data=data,
                    events=deepcopy(self.events), message=self.message)
