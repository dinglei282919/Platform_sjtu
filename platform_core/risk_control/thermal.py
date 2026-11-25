"""Common jacketed CSTR integration with an optional physical vessel heat load."""
import numpy as np
from numba import njit

from . import cstr


@njit(cache=True)
def plant_interval(x, flow, inlet, exchange, load, h):
    if load == 0.0:
        return cstr.integrate(x, flow, inlet, cstr.SAMPLE_MIN, h, 1.0, exchange)
    # The extra thermal calibration load is applied only in the vessel balance.
    added = np.array([0.0, load, 0.0])
    steps = round(cstr.SAMPLE_MIN / h)
    y = x.copy()
    for _ in range(steps):
        k1 = cstr.derivative(y, flow, inlet, 1.0, exchange) + added
        k2 = cstr.derivative(y + h*k1/2, flow, inlet, 1.0, exchange) + added
        k3 = cstr.derivative(y + h*k2/2, flow, inlet, 1.0, exchange) + added
        k4 = cstr.derivative(y + h*k3, flow, inlet, 1.0, exchange) + added
        y += h*(k1 + 2*k2 + 2*k3 + k4)/6
    return y
