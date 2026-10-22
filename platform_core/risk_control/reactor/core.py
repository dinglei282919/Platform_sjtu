"""Literal translation of the connected blocks in switch_sim.mdl (R2023a)."""
from dataclasses import asdict, dataclass
import math
import time
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

SWITCHES = (0.2, 0.7, 1.2, 1.6, 1.9, 2.3)
MODES = (2, 1, 2, 1, 2, 1, 2)
CENTERS1 = np.arange(-3, 4, dtype=float) * 5
# MATLAB d2(i), i=1:7, indexes the 2x7 matrix in column-major order.
CENTERS2 = np.repeat(CENTERS1, 2)[:7]


@dataclass(frozen=True)
class Config:
    duration: float = 3.0
    sample_dt: float = 0.002
    prescribed_time: float = 1.5
    ell: float = 0.15
    x1: float = 0.5
    x2: float = 0.5
    observer1: float = 0.0
    observer2: float = 0.5
    rtol: float = 1e-3
    atol: float = 1e-6
    method: str = "BDF"

    def validate(self):
        for name, value in asdict(self).items():
            if name != "method" and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.duration <= 3:
            raise ValueError("duration must be in (0, 3], the source-model time range")
        if not 0.0001 <= self.sample_dt <= self.duration:
            raise ValueError("sample_dt must be between 0.0001 and duration")
        if not 0 < self.prescribed_time <= self.duration or not 0 < self.ell:
            raise ValueError("require 0 < prescribed_time <= duration and ell > 0")
        if self.prescribed_time / self.ell > 20:
            raise ValueError("prescribed_time / ell must be <= 20 for numerical conditioning")
        if not 1e-10 <= self.rtol <= 1e-2 or not 1e-12 <= self.atol <= 1e-3:
            raise ValueError("rtol or atol outside supported range")
        if self.method not in ("Radau", "BDF", "LSODA"):
            raise ValueError("method must be Radau, BDF or LSODA")


class SimulationCancelled(Exception):
    pass


def switching_mode(t):
    # signal.m gives mode 1 at exactly 2.3; right limit is mode 2.
    if t == 2.3:
        return 1
    return MODES[int(np.searchsorted(SWITCHES, t, side="right"))]


def gamma(t, config):
    T, ell = config.prescribed_time, config.ell
    return math.exp(t / (T - t + ell)) if t <= T else math.exp(T / ell)


def nussbaum(eta):
    if np.isscalar(eta):
        invalid = not np.isfinite(eta) or abs(eta) > 26
    else:
        invalid = not np.all(np.isfinite(eta)) or np.any(np.abs(eta) > 26)
    if invalid:
        raise FloatingPointError("Nussbaum exponent exceeded numerical range; simulation stopped without clipping")
    return np.exp(eta * eta) * np.cos(math.pi * eta / 2)


def basis1(x):
    if np.isscalar(x):
        return np.exp(-((x - CENTERS1) ** 2) / 50)
    x = np.asarray(x)
    centers = CENTERS1.reshape((7,) + (1,) * x.ndim)
    return np.exp(-((x - centers) ** 2) / 50)


def basis2(x1, x2):
    if np.isscalar(x1):
        return np.exp(-((x1 - CENTERS2) ** 2 + (x2 - CENTERS2) ** 2) / 50)
    x1 = np.asarray(x1)
    centers = CENTERS2.reshape((7,) + (1,) * x1.ndim)
    return np.exp(-((x1 - centers) ** 2 + (x2 - centers) ** 2) / 50)


def initial_state(c):
    y = np.zeros(37)
    y[:4] = (c.x1, c.x2, c.observer1, c.observer2)
    y[4:18] = 0.5
    y[19:33] = 1
    y[36] = 0.75
    return y


def signals(t, y, c, mode=None):
    mode = switching_mode(t) if mode is None else mode
    g = gamma(t, c)
    k = 1.5 * math.sin(0.25 * t) + 2
    kx1, kx2 = k * y[0], k * y[3]
    v1, v2 = g * kx1, g * (kx2 - y[34])
    weight_shape = (2, 7) + y.shape[1:]
    th1 = y[4:18].reshape(weight_shape)[mode - 1]
    th2 = y[19:33].reshape(weight_shape)[mode - 1]
    h1, h2 = basis1(kx1), basis2(kx1, kx2)
    dot1 = th1 @ h1 if y.ndim == 1 else np.sum(th1 * h1, axis=0)
    dot2 = th2 @ h2 if y.ndim == 1 else np.sum(th2 * h2, axis=0)
    lam1 = v1 * dot1 + 1.25 * v1**3
    lam2 = v2 * dot2 + 0.5 * v2 + 1.5 * v2**3
    alpha1 = -9 * v1 + nussbaum(y[18]) * lam1
    alpha2 = -(19 + 0.5 * kx2**2) * v2 + nussbaum(y[33]) * lam2
    u = nussbaum(y[36]) * y[35] * v2 * alpha2**2
    return dict(gamma=g, k=k, mode=mode, kx1=kx1, kx2=kx2, v1=v1, v2=v2,
                theta1=th1, theta2=th2, h1=h1, h2=h2, lam1=lam1, lam2=lam2,
                alpha1=alpha1, alpha2=alpha2, u=u)


def derivatives(t, y, c, mode):
    with np.errstate(over="raise", invalid="raise"):
        s = signals(t, y, c, mode)
        g, v1, v2, a2 = s["gamma"], s["v1"], s["v2"], s["alpha2"]
        dy = np.zeros_like(y)
        w1, w2, decay1, decay2 = (0.75, 2, 0.2, 0.3) if mode == 1 else (1, 1.8, 0.4, 0.3)
        dy[0] = y[1] - (1 / w1 + decay1) * y[0]
        dy[1] = s["u"] - (1 / w2 + decay2 - 0.2 / 0.5) * y[1] + 1.2 * y[0]
        e = y[0] - y[2]
        hobs1, hobs2 = basis1(y[0]), basis2(y[2], y[3])
        ef1 = hobs1 @ s["theta1"] if y.ndim == 1 else np.sum(hobs1 * s["theta1"], axis=0)
        ef2 = hobs2 @ s["theta2"] if y.ndim == 1 else np.sum(hobs2 * s["theta2"], axis=0)
        dy[2] = y[3] + ef1 + g * 5 * e
        dy[3] = s["u"] + ef2 + g * 8 * e
        weight_shape = (2, 7) + y.shape[1:]
        dth1 = dy[4:18].reshape(weight_shape)
        dth1[:] = -3 * g * y[4:18].reshape(weight_shape)
        dth1[mode - 1] += g * v1**2 * s["h1"]
        dy[18] = v1 * s["lam1"]
        dth2 = dy[19:33].reshape(weight_shape)
        dth2[:] = -3 * g * y[19:33].reshape(weight_shape)
        dth2[mode - 1] += g * v2**2 * s["h2"]
        dy[33] = v2 * s["lam2"]
        dy[34] = g * (s["alpha1"] - y[34]) / 0.15
        dy[35] = g * v2**2 * a2**2 - 30 * g * y[35]
        dy[36] = y[35] * v2**2 * a2**2
        if not np.isfinite(dy).all():
            raise FloatingPointError("Non-finite derivative; no partial result exported")
        return dy


def jacobian(t, y, c, mode):
    """Complex-step Jacobian avoids finite-difference cancellation at high gamma."""
    perturbation = 1e-25
    states = np.repeat(y[:, None], 37, axis=1).astype(complex)
    states[np.diag_indices(37)] += 1j * perturbation
    return derivatives(t, states, c, mode).imag / perturbation


def simulate(config=None, progress: Callable = None, cancelled: Callable = None, *, compiled=True):
    c = config or Config()
    c.validate()
    if compiled:
        from .kernel import derivatives_compiled
        evaluate = lambda t, state, mode: derivatives_compiled(t, state, c.prescribed_time, c.ell, mode)
    else:
        evaluate = lambda t, state, mode: derivatives(t, state, c, mode)
    y = initial_state(c)
    boundaries = sorted({0.0, c.duration, *[t for t in SWITCHES if t < c.duration],
                         *([c.prescribed_time] if c.prescribed_time < c.duration else [])})
    grid = np.unique(np.r_[np.arange(0, c.duration, c.sample_dt), boundaries])
    values = np.empty((37, len(grid)))
    evaluations = 0
    started = time.monotonic()
    last_update = started
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mode = switching_mode((start + end) / 2)

        def rhs(t, state):
            nonlocal last_update
            if cancelled and cancelled():
                raise SimulationCancelled("Simulation cancelled")
            now = time.monotonic()
            if now - started > 1200:
                raise RuntimeError("Integration exceeded 20 minutes; no incomplete result exported")
            if progress and now - last_update > 0.5:
                progress(t / c.duration)
                last_update = now
            return evaluate(t, state, mode)

        sol = solve_ivp(rhs, (start, end), y, method=c.method, rtol=c.rtol, atol=c.atol,
                        dense_output=True, max_step=0.02,
                        jac=lambda t, state: jacobian(t, state, c, mode))
        if not sol.success:
            raise RuntimeError(f"Integration failed at {sol.t[-1]:.8g}: {sol.message}")
        mask = (grid >= start) & (grid <= end)
        values[:, mask] = sol.sol(grid[mask])
        y = sol.y[:, -1]
        evaluations += sol.nfev
        if progress:
            progress(end / c.duration)
    if cancelled and cancelled():
        raise SimulationCancelled("Simulation cancelled")
    extra = [signals(float(t), values[:, i], c) for i, t in enumerate(grid)]
    data = {"t": grid, "x1": values[0], "x2": values[1], "observer1": values[2],
            "observer2": values[3], "error1": values[0] - values[2],
            "error2": values[1] - values[3]}
    for name in ("u", "gamma", "k", "mode", "v1", "v2", "alpha1", "alpha2"):
        data[name] = np.array([s[name] for s in extra])
    for name in ("theta1", "theta2"):
        data[name + "_norm"] = np.array([np.linalg.norm(s[name]) for s in extra])
    if not all(np.isfinite(a).all() for a in data.values()):
        raise FloatingPointError("Non-finite result")
    metrics = {"final_x_norm": float(np.linalg.norm(y[:2])),
               "final_observer_error": float(np.linalg.norm(y[:2] - y[2:4])),
               "peak_abs_u_sampled": float(np.max(np.abs(data["u"]))),
               "nfev": evaluations, "samples": len(grid),
               "elapsed_seconds": time.monotonic() - started}
    return {"config": asdict(c), "data": data, "states": values, "metrics": metrics}
