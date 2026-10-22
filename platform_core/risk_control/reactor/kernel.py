"""Compiled scalar RHS; core.derivatives remains the independent reference.

No fastmath: preserve IEEE arithmetic and fail on overflow instead of clipping.
Compilation is cached on disk and happens in the simulation worker.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _nussbaum(eta):
    if not np.isfinite(eta) or abs(eta) > 26:
        raise FloatingPointError("Nussbaum exponent exceeded numerical range; simulation stopped without clipping")
    return np.exp(eta * eta) * np.cos(math.pi * eta / 2)


@njit(cache=True, nogil=True)
def derivatives_compiled(t, y, prescribed_time, ell, mode):
    g = math.exp(t / (prescribed_time - t + ell)) if t <= prescribed_time else math.exp(prescribed_time / ell)
    k = 1.5 * math.sin(0.25 * t) + 2
    kx1, kx2 = k * y[0], k * y[3]
    v1, v2 = g * kx1, g * (kx2 - y[34])
    h1, h2 = np.empty(7), np.empty(7)
    obs1, obs2 = np.empty(7), np.empty(7)
    for i in range(7):
        center1 = (i - 3) * 5.
        center2 = (i // 2 - 3) * 5.
        h1[i] = np.exp(-((kx1 - center1) ** 2) / 50)
        h2[i] = np.exp(-((kx1 - center2) ** 2 + (kx2 - center2) ** 2) / 50)
        obs1[i] = np.exp(-((y[0] - center1) ** 2) / 50)
        obs2[i] = np.exp(-((y[2] - center2) ** 2 + (y[3] - center2) ** 2) / 50)
    first = 4 + (mode - 1) * 7
    second = 19 + (mode - 1) * 7
    th1, th2 = y[first:first + 7], y[second:second + 7]
    lam1 = v1 * (th1 @ h1) + 1.25 * v1**3
    lam2 = v2 * (th2 @ h2) + 0.5 * v2 + 1.5 * v2**3
    alpha1 = -9 * v1 + _nussbaum(y[18]) * lam1
    alpha2 = -(19 + 0.5 * kx2**2) * v2 + _nussbaum(y[33]) * lam2
    u = _nussbaum(y[36]) * y[35] * v2 * alpha2**2
    dy = np.empty_like(y)
    w1, w2, decay1, decay2 = (0.75, 2., 0.2, 0.3) if mode == 1 else (1., 1.8, 0.4, 0.3)
    dy[0] = y[1] - (1 / w1 + decay1) * y[0]
    dy[1] = u - (1 / w2 + decay2 - 0.2 / 0.5) * y[1] + 1.2 * y[0]
    e = y[0] - y[2]
    dy[2] = y[3] + obs1 @ th1 + g * 5 * e
    dy[3] = u + obs2 @ th2 + g * 8 * e
    for i in range(14):
        dy[4 + i] = -3 * g * y[4 + i]
        dy[19 + i] = -3 * g * y[19 + i]
    for i in range(7):
        dy[first + i] += g * v1**2 * h1[i]
        dy[second + i] += g * v2**2 * h2[i]
    dy[18] = v1 * lam1
    dy[33] = v2 * lam2
    dy[34] = g * (alpha1 - y[34]) / 0.15
    dy[35] = g * v2**2 * alpha2**2 - 30 * g * y[35]
    dy[36] = y[35] * v2**2 * alpha2**2
    if not np.isfinite(dy).all():
        raise FloatingPointError("Non-finite derivative; no partial result exported")
    return dy
