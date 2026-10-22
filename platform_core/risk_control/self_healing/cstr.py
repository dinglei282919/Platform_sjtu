"""Deterministic CSTR benchmark ported from the supplied Simulink models.

Source: ``Self-healing control-多变量非线性过程相关性自愈控制-CSTR/SCSC/``
``cstr_feedback_anormal.slx`` and the corresponding fault/self models. Their
``simulink/blockdiagram.xml`` stores the three balance equations, PI parameters,
initial states, and fault profiles. Time uses the original model's units.

The default interface follows the paper's three manipulated inputs. The legacy
interface also permits changing the reactor-temperature setpoint. In the MATLAB
script column 11 of ``simout`` is the actual reactor temperature, although its
correction is applied to the temperature setpoint; that ambiguity is not copied.

Engineering additions are explicit: bounded supervisory inputs, conditional PI
anti-windup (can be disabled), deterministic integration, and finite-value checks.
The legacy Simulink faults start at 200, whereas the paper's prose uses 160 in
places; ``fault_start`` makes the choice explicit. No sensor/process noise is
silently added. Measurement noise belongs in the experiment runner.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


class CSTRPlant:
    """Three-state stirred tank with a local PI cooling-flow controller.

    ``step(u, t)`` holds ``u`` for ``dt`` model-time units, integrates from
    ``t`` to ``t + dt``, and returns the state at the interval end. Calls must
    use consecutive times; training and evaluation can restart with ``reset``.

    Source PI: error = T - Tsp, Kp = 1, Ki = 5, Qc in [10, 200]. The
    integrator initially contributes 148.550238092053. Integration uses RK4
    with substeps no greater than 0.01 for the fast jacket-temperature state.
    """

    output_names = ("C", "T", "Tc")
    reference_output = np.array([0.1, 430.882699002514, 416.723084574301])
    nominal_cooling_flow = 148.550238092053
    cooling_flow_limits = (10.0, 200.0)

    def __init__(
        self,
        dt: float = 1.0,
        fault: Literal["none", "htc", "catalyst"] = "none",
        fault_start: float = 200.0,
        fault_rate: float | None = None,
        input_mode: Literal["paper", "legacy"] = "paper",
        integration_step: float = 0.01,
        anti_windup: bool = True,
        catalyst_initial_factor: float = 1.0,
    ) -> None:
        if fault not in ("none", "htc", "catalyst"):
            raise ValueError("fault must be 'none', 'htc', or 'catalyst'")
        if input_mode not in ("paper", "legacy"):
            raise ValueError("input_mode must be 'paper' or 'legacy'")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(integration_step) or not 0 < integration_step <= 0.01:
            raise ValueError("integration_step must be in (0, 0.01]")
        if not math.isfinite(fault_start) or fault_start < 0:
            raise ValueError("fault_start must be finite and nonnegative")
        if not math.isfinite(catalyst_initial_factor) or not 0 < catalyst_initial_factor <= 1:
            raise ValueError("catalyst_initial_factor must be in (0, 1]")
        if fault_rate is None:
            fault_rate = 0.001 if fault == "htc" else 0.0005
        if not math.isfinite(fault_rate) or fault_rate < 0:
            raise ValueError("fault_rate must be finite and nonnegative")
        self.dt = float(dt)
        self.fault = fault
        self.fault_start = float(fault_start)
        self.fault_rate = float(fault_rate)
        self.input_mode = input_mode
        self.integration_step = float(integration_step)
        self.anti_windup = bool(anti_windup)
        self.catalyst_initial_factor = float(catalyst_initial_factor)
        self.input_names = ("Ci", "Ti", "Tci")
        self.nominal_input = np.array([1.0, 350.0, 350.0])
        # Supervisory actuator limits are engineering assumptions, not paper data.
        lower = [0.2, 280.0, 250.0]
        upper = [2.0, 420.0, 410.0]
        if input_mode == "legacy":
            self.input_names += ("Tsp",)
            self.nominal_input = np.append(self.nominal_input, self.reference_output[1])
            lower.append(390.0)
            upper.append(470.0)
        self.input_bounds = (np.array(lower), np.array(upper))
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the process and PI integrator to the nominal equilibrium."""
        self._state = tuple(float(v) for v in self.reference_output) + (self.nominal_cooling_flow,)
        self.time = 0.0
        self.last_input = self.nominal_input.copy()
        self.cooling_flow = self.nominal_cooling_flow
        return self.state

    @property
    def state(self) -> np.ndarray:
        """Return a copy of [concentration, reactor T, jacket T]."""
        return np.array(self._state[:3])

    def fault_factors(self, time: float) -> tuple[float, float]:
        """Return (catalyst activity, heat-transfer factor) at ``time``.

        Source exp[-rate*(t-200)] passes through a [0,1] saturation. A
        non-unit ``catalyst_initial_factor`` explicitly models initial damage.
        """
        elapsed = max(0.0, time - self.fault_start)
        factor = math.exp(-self.fault_rate * elapsed)
        catalyst = self.catalyst_initial_factor * factor if self.fault == "catalyst" else 1.0
        htc = factor if self.fault == "htc" else 1.0
        return catalyst, htc

    def _rhs(self, time: float, state: tuple[float, ...], inputs: tuple[float, ...]) -> tuple[float, ...]:
        c, temp, jacket, integral = state
        ci, ti, tci, tsp = inputs
        if temp <= 0 or not all(math.isfinite(v) for v in state):
            raise FloatingPointError("CSTR integration left the finite positive-temperature domain")
        catalyst, htc = self.fault_factors(time)
        reaction = 7.2e10 * c * catalyst * math.exp(-1.0e4 / temp)
        error = temp - tsp
        raw_flow = error + integral
        flow = min(200.0, max(10.0, raw_flow))
        integral_rate = 5.0 * error
        if self.anti_windup and ((raw_flow >= 200.0 and integral_rate > 0) or (raw_flow <= 10.0 and integral_rate < 0)):
            integral_rate = 0.0
        return (
            (ci - c) / 1.5 - reaction,
            (ti - temp) / 1.5 + 200.0 * reaction - (7.0 / 1.5) * htc * (temp - jacket),
            flow / 10.0 * (tci - jacket) + 70.0 * htc * (temp - jacket),
            integral_rate,
        )

    def step(self, setpoints: np.ndarray, time: float) -> np.ndarray:
        """Advance one sample; clip setpoints to the documented input bounds."""
        command = np.asarray(setpoints, dtype=float)
        if command.shape != self.nominal_input.shape or not np.isfinite(command).all():
            raise ValueError(f"setpoints must be finite with shape {self.nominal_input.shape}")
        if not math.isfinite(time) or not math.isclose(time, self.time, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"time must equal the next interval start {self.time:g}")
        command = np.clip(command, *self.input_bounds)
        inputs = tuple(float(v) for v in command)
        if self.input_mode == "paper":
            inputs += (float(self.reference_output[1]),)
        count = math.ceil(self.dt / self.integration_step)
        h = self.dt / count
        state = self._state
        for index in range(count):
            t = time + index * h
            k1 = self._rhs(t, state, inputs)
            k2 = self._rhs(t + h / 2, tuple(s + h * k / 2 for s, k in zip(state, k1)), inputs)
            k3 = self._rhs(t + h / 2, tuple(s + h * k / 2 for s, k in zip(state, k2)), inputs)
            k4 = self._rhs(t + h, tuple(s + h * k for s, k in zip(state, k3)), inputs)
            state = tuple(s + h * (a + 2 * b + 2 * c + d) / 6 for s, a, b, c, d in zip(state, k1, k2, k3, k4))
        if not all(math.isfinite(v) for v in state) or min(state[:3]) <= 0:
            raise FloatingPointError("CSTR integration produced a nonphysical state; reduce integration_step")
        self._state = state
        self.time = float(time) + self.dt
        self.last_input = command.copy()
        self.cooling_flow = float(np.clip(state[1] - inputs[3] + state[3], 10.0, 200.0))
        return self.state
