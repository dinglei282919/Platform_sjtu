"""Published eligibility criteria and matched independent mechanistic closed-loop trials."""
import time
import numpy as np
from ..cstr import INLET, NOMINAL, NOMINAL_FLOW, SAMPLE_MIN, step
from .mpc import Settings, solve
from .training import errors

ONE_LIMIT = np.array([0.008, 0.40, 0.60])
ROLLOUT_LIMIT = np.array([0.025, 1.15, 1.3])


def closed_loop(model, settings, *, controller="dnn", initial=None, inlet=INLET,
                offsets=(1.5, -1.0), segment=24, cancelled=None):
    x = np.array(NOMINAL if initial is None else initial, dtype=float)
    flow = NOMINAL_FLOW
    integral = flow
    warm = None
    series, durations = [], []
    for delta in offsets:
        ref = float(np.clip(NOMINAL[1]+delta, 428, 434))
        for _ in range(segment):
            if cancelled and cancelled():
                raise InterruptedError("任务已取消")
            start = time.perf_counter()
            if controller == "pi":
                error = x[1]-ref
                requested = error+integral
                flow = float(np.clip(requested, settings.flow_min, settings.flow_max))
                if flow == requested or (requested-flow)*error < 0:
                    integral += SAMPLE_MIN*5*error
            else:
                warm, _ = solve(None if controller == "mechanistic" else model, x, flow, ref,
                                settings, warm=warm, inlet=inlet)
                flow = float(warm[0])
            x = step(x, flow, inlet)
            durations.append(1000*(time.perf_counter()-start))
            series.append([*x, flow, ref])
    data = np.array(series)
    err = data[:, 1]-data[:, 4]
    summary = dict(rmse_K=float(np.sqrt(np.mean(err**2))), iae_K_min=float(abs(err).sum()*SAMPLE_MIN),
                   last_error_K=float(err[-1]), concentration_deviation=float(max(abs(data[:, 0]-NOMINAL[0]))),
                   flow_min=float(min(data[:, 3])), flow_max=float(max(data[:, 3])),
                   total_flow_change=float(abs(np.diff(data[:, 3], prepend=NOMINAL_FLOW)).sum()),
                   p95_ms=float(np.percentile(durations, 95)), max_ms=float(max(durations)),
                   misses=int(sum(v > 60000*SAMPLE_MIN/settings.speed for v in durations)),
                   cycles=len(durations))
    return summary, series


def check(model, validation, settings, *, active=None, recent=None, cancelled=None):
    """Test data is reserved for the final report. Candidates see validation data only."""
    report = {"prediction": errors(model, validation, settings.horizon)}
    evaluations = [("nominal", None, INLET)]
    if recent is not None:
        report["recent_prediction"] = errors(model, recent, settings.horizon)
        evaluations.append(("recent", recent.features[-1, 0, :3], recent.features[-1, 0, 4:]))
    reasons = []
    for key in ("prediction", "recent_prediction"):
        if key not in report:
            continue
        for name, bound in (("one_rmse", ONE_LIMIT), ("rollout_rmse", ROLLOUT_LIMIT)):
            metric = np.asarray(report[key][name])
            if not np.isfinite(metric).all() or (metric > bound).any():
                reasons.append(f"{key}/{name} 超出预设阈值")
            if active is not None:
                source = recent if key.startswith("recent") else validation
                reference = np.asarray(errors(active, source, settings.horizon)[name])
                if (metric > reference*1.25 + 0.05*bound).any():
                    reasons.append(f"{key}/{name} 相对活动模型退化")
    for name, x, d in evaluations:
        metrics, _ = closed_loop(model, settings, initial=x, inlet=d, segment=12, cancelled=cancelled)
        report[name] = metrics
        if (metrics["rmse_K"] > 1.4 or metrics["misses"] or metrics["flow_min"] < settings.flow_min-1e-6
                or metrics["flow_max"] > settings.flow_max+1e-6):
            reasons.append(f"{name} 闭环误差、流量约束或周期时限不合格")
        if active is not None:
            old, _ = closed_loop(active, settings, initial=x, inlet=d, segment=12, cancelled=cancelled)
            report[name+"_active"] = old
            if metrics["rmse_K"] > 1.15*old["rmse_K"]+0.04:
                reasons.append(f"{name} 闭环跟踪相对活动模型退化")
    return {"passed": not reasons, "reasons": reasons, **report}
