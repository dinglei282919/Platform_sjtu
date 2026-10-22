"""Stiff integration with committed major-step derivative history."""
import numpy as np
from scipy.integrate import BDF
from .config import validate_config
from .model import FdiModel, SWITCH_TIMES, initial_state

SOLVER_SETTINGS = {"rtol": 1e-6, "atol": 1e-8, "max_step": .001}


class SimulationCancelled(RuntimeError):
    pass


def simulate(config=None, *, progress=None, cancelled=None, on_snapshot=None):
    duration = validate_config(config or {})["sim_time"]
    model = FdiModel()
    state = initial_state()
    rows = [model.evaluate(0., state)[1]]
    states = [state.copy()]
    model.accept(0., 1.)
    start = 0.
    last_progress = -1
    # Restart at known discontinuities instead of allowing a large solver
    # step to skip a switching interval. State and derivative history persist.
    ends = list(SWITCH_TIMES[SWITCH_TIMES < duration]) + [duration]
    for end in ends:
        solver = BDF(model.rhs, start, state, end, **SOLVER_SETTINGS)
        while solver.status == "running":
            if cancelled is not None and cancelled():
                raise SimulationCancelled("仿真已取消。")
            message = solver.step()
            if solver.status == "failed":
                raise RuntimeError("FDI 求解器失败：" + str(message))
            _, row = model.evaluate(solver.t, solver.y)
            if not np.isfinite(solver.y).all() or solver.y[4] < -1e-7:
                raise RuntimeError("FDI 状态非有限或饱和补偿状态越界。")
            if not 0 < row["eta1"] < 1 or any(not np.isfinite(v) for v in row.values()):
                raise RuntimeError("FDI 性能边界或输出信号越界。")
            rows.append(row)
            states.append(solver.y.copy())
            model.accept(solver.t, row["eta1"])
            percent = int(100 * solver.t / duration)
            if percent != last_progress:
                if on_snapshot is not None:
                    on_snapshot(float(solver.t), duration,
                                lambda: {key: np.array([row[key] for row in rows]) for key in rows[0]})
                if progress is not None:
                    progress(float(solver.t), duration)
                last_progress = percent
            if len(rows) > 200000:
                raise RuntimeError("FDI 求解步数超过限制，请检查模型参数。")
        start, state = solver.t, solver.y.copy()
    return {key: np.array([row[key] for row in rows]) for key in rows[0]} | {"states": np.array(states)}
