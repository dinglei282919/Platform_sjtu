"""A measured inlet pulse and a matched constant-flow comparison for a live CSTR run."""
from dataclasses import dataclass

from .cstr import INLET, SAMPLE_MIN, step


@dataclass(frozen=True)
class InletPreset:
    label: str
    index: int
    target: float
    unit: str

    def entry(self, identifier):
        return dict(id=identifier, label=self.label, nominal=float(INLET[self.index]),
                    target=self.target, unit=self.unit)


PRESETS = {
    "feed_concentration": InletPreset("进料浓度升高", 0, 1.03, "mol/L"),
    "feed_temperature": InletPreset("进料温度升高", 1, 352.0, "K"),
    "coolant_temperature": InletPreset("冷却液入口降温", 2, 348.0, "K"),
}
DEFAULT_CYCLES = 20
MAX_CYCLES = 60
RECOVERY_CYCLES = 10


def options():
    return dict(presets=[p.entry(k) for k, p in PRESETS.items()], default_cycles=DEFAULT_CYCLES,
                max_cycles=MAX_CYCLES, recovery_cycles=RECOVERY_CYCLES)


class InletTrial:
    def __init__(self, key, preset, cycles):
        if not isinstance(preset, str) or preset not in PRESETS:
            raise ValueError("请选择有效的入口扰动")
        if type(cycles) is not int or not 1 <= cycles <= MAX_CYCLES:
            raise ValueError(f"扰动持续周期须为 1–{MAX_CYCLES} 的整数")
        self.key, self.preset, self.cycles = key, preset, cycles
        definition = PRESETS[preset]
        self.input = INLET.copy()
        self.input[definition.index] = definition.target
        self.phase = "queued"
        self.clear_requested = False
        self.start_seq = self.release_seq = None
        self.held_state = self.held_flow = None
        self.closed_peak = self.held_peak = None

    @property
    def observing(self):
        return self.phase in ("active", "recovery")

    def clear(self):
        if self.phase == "queued":
            self.phase = "cancelled"
        elif self.phase == "active":
            self.clear_requested = True
        else:
            raise ValueError("当前没有待撤除的扰动")

    def boundary(self, seq, state, flow, reference):
        if self.phase == "queued":
            self.phase, self.start_seq = "active", seq
            self.held_state, self.held_flow = state.copy(), float(flow)
            self.closed_peak = self.held_peak = abs(float(state[1])-reference)
            return "disturbance_applied"
        if self.phase == "active" and (self.clear_requested or seq >= self.start_seq + self.cycles):
            self.phase, self.release_seq = "recovery", seq
            return "disturbance_restored"
        return None

    def compare(self, inlet):
        if self.held_state is not None:
            try:
                return step(self.held_state, self.held_flow, inlet)
            except (ValueError, FloatingPointError, OverflowError):
                # A failed comparison must not interrupt the controlled plant.
                return None
        return None

    def sample_fields(self, comparison):
        return dict(trial_id=self.key, disturbance_id=self.preset, disturbance_phase=self.phase,
                    T_hold=float(comparison[1]) if comparison is not None else None, Qc_hold=self.held_flow)

    def commit(self, seq, temp, reference, comparison):
        self.held_state = comparison
        self.closed_peak = max(self.closed_peak, abs(float(temp)-reference))
        self.held_peak = (max(self.held_peak, abs(float(comparison[1])-reference))
                          if comparison is not None else None)
        if self.phase == "recovery" and seq >= self.release_seq + RECOVERY_CYCLES:
            self.phase = "complete"
            return True
        return False

    def summary(self, seq):
        remaining = 0
        if self.phase == "active":
            remaining = max(0, self.start_seq + self.cycles - seq)
        elif self.phase == "recovery":
            remaining = max(0, self.release_seq + RECOVERY_CYCLES - seq)
        return dict(trial_id=self.key, preset=self.preset, phase=self.phase, cycles=self.cycles,
                    remaining=remaining, clear_requested=self.clear_requested,
                    start_min=None if self.start_seq is None else self.start_seq*SAMPLE_MIN,
                    restore_min=None if self.release_seq is None else self.release_seq*SAMPLE_MIN,
                    held_flow=self.held_flow, closed_peak_K=self.closed_peak, held_peak_K=self.held_peak)

