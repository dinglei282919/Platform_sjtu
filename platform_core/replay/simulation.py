"""Method of steps with dense past-state history for the transport delays."""
from bisect import bisect_right
import numpy as np
from scipy.integrate import solve_ivp
from .config import ATTACKS, attack_index, validate_config
from .model import INITIAL_STATE, evaluate

SOLVER_SETTINGS = dict(rtol=2e-7, atol=1e-9, max_step=.02)


class SimulationCancelled(RuntimeError):
    pass


class History:
    def __init__(self):
        self.starts, self.solutions = [], []

    def append(self, start, solution):
        self.starts.append(start)
        self.solutions.append(solution)

    def at(self, t):
        if t <= 0:
            return INITIAL_STATE[:2].copy()
        index = bisect_right(self.starts, t)-1
        if index < 0 or t > self.solutions[index].t[-1]+1e-10:
            raise ValueError("重放历史尚不可用。")
        return self.solutions[index].sol(t)[:2]


def simulate(config=None, *, progress=None, cancelled=None, solver_settings=None, on_snapshot=None):
    end = validate_config(config or {})["sim_time"]
    settings = SOLVER_SETTINGS | (solver_settings or {})
    history = History()
    state = INITIAL_STATE.copy()
    preview_times = np.unique(np.r_[np.arange(0., end, .01), end]) if on_snapshot is not None else None
    preview_data, preview_count = None, 0

    def snapshot(stop):
        nonlocal preview_data, preview_count
        count = int(np.searchsorted(preview_times, stop, side="right"))
        if count > preview_count:
            # Read each completed sample only once for previews. The final
            # export still samples the complete history independently below.
            chunk = _sample_history(history, stop, cancelled, times=preview_times[preview_count:count])
            preview_data = chunk if preview_data is None else {
                key: np.concatenate((preview_data[key], chunk[key])) for key in chunk}
            preview_count = count
        return preview_data
    # Each interval is shorter than the minimum delay (2.5 s), so RHS
    # evaluations can only read completed history, never tentative steps.
    boundaries = np.unique(np.r_[np.arange(0, end, .5), end,
                                  [v for a in ATTACKS for v in a[:2] if 0 < v < end]])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        active = attack_index((start+stop)/2)
        delay = ATTACKS[active-1][2] if active else 0.

        def rhs(t, x):
            if cancelled is not None and cancelled():
                raise SimulationCancelled("仿真已取消。")
            received = history.at(t-delay) if active else x[:2]
            # Both cascaded Simulink Derivative blocks initially output zero.
            return evaluate(t, x, received, reference_values=(0., 0., 0.) if t == 0 else None)[0]

        solution = solve_ivp(rhs, (start, stop), state, method="BDF", dense_output=True, **settings)
        if not solution.success or not np.all(np.isfinite(solution.y)):
            raise RuntimeError("重放仿真求解失败：" + solution.message)
        history.append(start, solution)
        state = solution.y[:, -1]
        if on_snapshot is not None:
            on_snapshot(float(stop), end, lambda: snapshot(stop))
        if progress is not None:
            progress(stop, end)
    return _sample_history(history, end, cancelled)


def _sample_history(history, end, cancelled=None, *, times=None):
    """Read only completed history for final output or a running preview."""
    if times is None:
        times = np.unique(np.r_[np.arange(0., end, .01), end])
    rows, states = [], []
    for t in times:
        if cancelled is not None and cancelled():
            raise SimulationCancelled("仿真已取消。")
        index = max(0, bisect_right(history.starts, t)-1)
        x = history.solutions[index].sol(t)
        active = attack_index(t)
        delay = ATTACKS[active-1][2] if active else 0.
        received = history.at(t-delay) if active else x[:2]
        _, row = evaluate(t, x, received, reference_values=(0., 0., 0.) if t == 0 else None)
        row.update(attack=active, delay=delay)
        rows.append(row)
        states.append(x)
    return {"t": times, "states": np.asarray(states),
            **{name: np.array([row[name] for row in rows]) for name in rows[0]}}
