"""Three water tanks, based on paper Eq. (32), with explicit engineering changes.

Source: ``Self-healing control-多变量非线性过程相关性自愈控制-JZSH/
SCSC/jinzhoushihua.slx``, Tank7--Tank9 and embedded MATLAB Functions.
The source uses nominal levels 50, outlet coefficients [5.657, 1.414, 1.4],
exp(k2*u + intercept) inlet characteristics, unit-area integrators, outer
level PI gains P=1, I=[0.5, 0.3, 0.2], and an inner flow PI P=50, I=200.
Its outlet-3 fault is a 0.1/day ramp beginning at day 40, subtracted from
the reported flow but NOT the tank's mass-balance feedback.

This portable model preserves dh/dt=(qin-qout)/A and qout=k3*sqrt(h).
It is an independently reproducible implementation, not a bit-for-bit SLX
reproduction. Engineering changes are intentional:

* All tanks start at the nominal steady state; random SLX noise is omitted.
* The inner flow loop is replaced by ideal inverse valve compensation.
* qin=exp(intercept)*expm1(k2*u), u in [0,100], permits actual shutoff.
  This translates the source exponential curve vertically to give qin(0)=0.
* The source outer PI gains are used with equilibrium feedforward and
  conditional integration anti-windup; they are not a retuned source PID.
* Blockage decreases the physical outlet-3 coefficient linearly to 40% over
  60 days, affecting BOTH outlet flow and mass balance. This approximates
  the source loss of 6 flow units at nominal q3=9.899 over days 40--100.
* Level setpoints [5,95] and physical tank levels [0,100] are engineering
  safety bounds, not paper-derived level constraints. Excess flow at the
  physical upper boundary is treated as spill, not useful outlet flow.

Time uses the paper's day coordinate, with default batch/sample dt=0.1.
RK4 integrates the three levels and three outer-PI integral states together.
"""

from __future__ import annotations

import math

import numpy as np


class WaterPlant:
    """Stateful water plant; ``step(u, t)`` advances [t, t+dt].

    ``u`` contains three level setpoints and the returned output is
    ``[total_flow, tank1_flow, tank2_flow, tank3_flow]`` at t+dt.
    The public ``state`` contains levels only. ``reset()`` restores both
    levels and PI states; the fault trajectory is determined by absolute t.
    """

    input_names = ("h1_setpoint", "h2_setpoint", "h3_setpoint")
    output_names = ("total_flow", "Q1", "Q2", "Q3")
    time_unit = "day"

    def __init__(
        self,
        dt: float = 0.1,
        fault: str = "none",
        fault_start: float = 40.0,
        *,
        fault_duration: float = 60.0,
        final_outlet_fraction: float = 0.4,
        max_internal_step: float = 0.02,
    ) -> None:
        for name, value in (("dt", dt), ("fault_duration", fault_duration),
                            ("max_internal_step", max_internal_step)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if fault not in {"none", "blockage"}:
            raise ValueError("fault must be 'none' or 'blockage'")
        if not np.isfinite(fault_start) or fault_start < 0:
            raise ValueError("fault_start must be finite and nonnegative")
        if not np.isfinite(final_outlet_fraction) or not 0 <= final_outlet_fraction <= 1:
            raise ValueError("final_outlet_fraction must lie in [0, 1]")
        self.dt = float(dt)
        self.fault = fault
        self.fault_start = float(fault_start)
        self.fault_duration = float(fault_duration)
        self.final_outlet_fraction = float(final_outlet_fraction)
        self.max_internal_step = float(max_internal_step)
        self.nominal_input = np.full(3, 50.0)
        self.initial_state = self.nominal_input.copy()
        self.input_bounds = (np.full(3, 5.0), np.full(3, 95.0))
        self.area = np.ones(3)
        self.outlet_coefficients = np.array([5.657, 1.414, 1.4])
        self.valve_slope = np.array([0.029, 0.0166, 0.0141])
        self.valve_scale = np.exp(np.array([2.0611, 2.0763, 2.1930]))
        self.kp = np.ones(3)
        self.ki = np.array([0.5, 0.3, 0.2])
        self._nominal_flow = self.outlet_coefficients * np.sqrt(self.initial_state)
        self._maximum_inlet = self.inlet_flow(np.full(3, 100.0))
        self.reference_output = np.r_[self._nominal_flow.sum(), self._nominal_flow]
        self.reset()

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length 3")
        return result

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """Reset to nominal operation, optionally supplying physical levels."""
        levels = self.initial_state if state is None else self._vector(state, "state")
        if np.any(levels < 0) or np.any(levels > 100):
            raise ValueError("initial physical levels must lie in [0, 100]")
        self.state = levels.copy()
        self._integral = np.zeros(3)
        self.time = 0.0
        self.last_setpoints = self.nominal_input.copy()
        self.valve_opening = np.log1p(self._nominal_flow / self.valve_scale) / self.valve_slope
        return self.output(0.0)

    def inlet_flow(self, valve_opening: np.ndarray) -> np.ndarray:
        """Physical shifted exponential valve flow, with 0--100% travel."""
        valve = self._vector(valve_opening, "valve_opening")
        return self.valve_scale * np.expm1(self.valve_slope * np.clip(valve, 0.0, 100.0))

    def outlet_fraction(self, time: float) -> float:
        """Tank-3 conductance relative to normal; the other tanks are healthy."""
        if self.fault == "none":
            return 1.0
        progress = np.clip((time - self.fault_start) / self.fault_duration, 0.0, 1.0)
        return float(1.0 - (1.0 - self.final_outlet_fraction) * progress)

    def flow_rates(self, time: float, levels: np.ndarray | None = None) -> np.ndarray:
        """Useful outlet flows at supplied physical levels (or current state)."""
        h = self.state if levels is None else self._vector(levels, "levels")
        flow = self.outlet_coefficients * np.sqrt(np.maximum(h, 0.0))
        flow[2] *= self.outlet_fraction(time)
        return flow

    def output(self, time: float | None = None) -> np.ndarray:
        flow = self.flow_rates(self.time if time is None else time)
        return np.r_[flow.sum(), flow]

    def derivative(self, levels: np.ndarray, inlet_flows: np.ndarray, time: float) -> np.ndarray:
        """Eq. (32) before the physical overflow boundary is applied."""
        h = self._vector(levels, "levels")
        inflow = self._vector(inlet_flows, "inlet_flows")
        return (inflow - self.flow_rates(time, h)) / self.area

    def _rhs(self, combined: np.ndarray, setpoints: np.ndarray, time: float) -> np.ndarray:
        levels, integral = combined[:3], combined[3:]
        error = setpoints - levels
        command = self._nominal_flow + self.kp * error + self.ki * integral
        inflow = np.clip(command, 0.0, self._maximum_inlet)
        # Conditional integration lets a saturated controller unwind.
        winding = ((command <= 0) & (error < 0)) | ((command >= self._maximum_inlet) & (error > 0))
        integral_dot = np.where(winding, 0.0, error)
        level_dot = (inflow - self.flow_rates(time, levels)) / self.area
        level_dot = np.where((levels >= 100.0) & (level_dot > 0), 0.0, level_dot)
        level_dot = np.where((levels <= 0.0) & (level_dot < 0), 0.0, level_dot)
        return np.r_[level_dot, integral_dot]

    def step(self, setpoints: np.ndarray, time: float) -> np.ndarray:
        """Integrate one sample, clipping level requests to engineering bounds."""
        requested = self._vector(setpoints, "setpoints")
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be finite and nonnegative")
        setpoints = np.clip(requested, *self.input_bounds)
        combined = np.r_[self.state, self._integral]
        count = max(1, int(math.ceil(self.dt / self.max_internal_step)))
        step_size = self.dt / count
        for index in range(count):
            t = time + index * step_size
            k1 = self._rhs(combined, setpoints, t)
            k2 = self._rhs(combined + 0.5 * step_size * k1, setpoints, t + 0.5 * step_size)
            k3 = self._rhs(combined + 0.5 * step_size * k2, setpoints, t + 0.5 * step_size)
            k4 = self._rhs(combined + step_size * k3, setpoints, t + step_size)
            combined += step_size / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            combined[:3] = np.clip(combined[:3], 0.0, 100.0)
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError("water integration produced non-finite state")
        self.state = combined[:3].copy()
        self._integral = combined[3:].copy()
        self.time = float(time + self.dt)
        self.last_setpoints = setpoints.copy()
        flow_command = self._nominal_flow + self.kp * (setpoints - self.state) + self.ki * self._integral
        inlet = np.clip(flow_command, 0.0, self._maximum_inlet)
        self.valve_opening = np.log1p(inlet / self.valve_scale) / self.valve_slope
        return self.output()
