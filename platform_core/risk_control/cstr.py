"""Pilario & Cao, Sec. IV-A, eqs. 26–28 and Table I. Time is in minutes."""
import math

import numpy as np
from numba import njit

SAMPLE_MIN = 0.1
INTEGRATION_MIN = 0.01
NOMINAL = np.array([0.1, 430.882699002514, 416.723084574301])
# Feed concentration and coolant flow are nominal. C and Tc balance at the initial T.
DEMO_INITIAL = np.array([0.094726, 432.0, 417.6448])
TREND_INTERVALS = 120
TREND_MAX_INTERVALS = 480
INLET = np.array([1.0, 350.0, 350.0])
NOMINAL_FLOW = 148.550238092053
FLOW_LIMITS = (10.0, 200.0)
REFERENCE_LIMITS = (428.0, 434.0)
STATE_NAMES = ("C", "T", "Tc")
FEATURE_NAMES = (*STATE_NAMES, "Qc", "Ci", "Ti", "Tci")
UNITS = ("mol/L", "K", "K", "L/min", "mol/L", "K", "K")
PLANT_VERSION = "pilario-cao-cstr-v1"


@njit(cache=True)
def derivative(state, flow, inlet, activity=1.0, heat_transfer=1.0):
    c, temp, jacket = state
    ci, ti, tci = inlet
    rate = activity * 7.2e10 * math.exp(-1.0e4 / temp) * c
    heat = heat_transfer * 7.0e5 * (temp - jacket)
    return np.array([(ci - c) * (100.0 / 150.0) - rate,
                     (ti - temp) * (100.0 / 150.0) + 200.0 * rate - heat / 150000.0,
                     flow / 10.0 * (tci - jacket) + heat / 10000.0])


@njit(cache=True)
def integrate(state, flow, inlet, period=SAMPLE_MIN, substep=INTEGRATION_MIN,
              activity=1.0, heat_transfer=1.0):
    """One zero-order-held input interval. No controller state lives in this function."""
    count = int(math.ceil(period / substep))
    h = period / count
    x = state.copy()
    for _ in range(count):
        a = derivative(x, flow, inlet, activity, heat_transfer)
        b = derivative(x + 0.5*h*a, flow, inlet, activity, heat_transfer)
        c = derivative(x + 0.5*h*b, flow, inlet, activity, heat_transfer)
        d = derivative(x + h*c, flow, inlet, activity, heat_transfer)
        x += h/6.0 * (a + 2*b + 2*c + d)
    return x


def apply_flow(command):
    if not math.isfinite(command):
        raise ValueError("冷却液流量必须为有限值")
    return float(np.clip(command, *FLOW_LIMITS))


def step(state, flow, inlet=INLET, period=SAMPLE_MIN, substep=INTEGRATION_MIN):
    x, d = np.asarray(state, dtype=float), np.asarray(inlet, dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() or min(v) <= 0 for v in (x, d)):
        raise ValueError("反应器状态和入口工况须含三个有限正值")
    if not (0 < period <= 1 and 0 < substep <= INTEGRATION_MIN):
        raise ValueError("采样周期或积分子步超出范围")
    result = integrate(x, apply_flow(flow), d, period, substep)
    if not np.isfinite(result).all() or min(result) <= 0:
        raise FloatingPointError("反应器积分越出物理域")
    return result
