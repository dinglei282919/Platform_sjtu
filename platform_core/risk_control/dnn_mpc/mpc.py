"""One SLSQP solve optimizes a sequence. Only the first accepted flow reaches the plant."""
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from numba import njit
from scipy.optimize import minimize
from ..cstr import INLET, NOMINAL, NOMINAL_FLOW, REFERENCE_LIMITS, SAMPLE_MIN, integrate

FLOW_SCALE = 20.0
TEMP_SCALE = 2.0


@dataclass(frozen=True)
class Settings:
    horizon: int = 30
    move_weight: float = 0.15
    flow_min: float = 100.0
    flow_max: float = 195.0
    reference: float = round(float(NOMINAL[1]), 3)
    speed: float = 4.0

    def __post_init__(self):
        if type(self.horizon) is not int or not 3 <= self.horizon <= 40:
            raise ValueError("预测时域应为 3–40 的整数")
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("参数须为有限数值")
        if not 95 <= self.flow_min <= NOMINAL_FLOW <= self.flow_max <= 200 or self.flow_min == self.flow_max:
            raise ValueError("流量范围须在 95–200 L/min 内并含名义流量")
        if not 0.005 <= self.move_weight <= 5 or not 0.25 <= self.speed <= 10:
            raise ValueError("流量变化权重或推进倍率越界")
        check_reference(self.reference)


def check_reference(value):
    if not math.isfinite(value) or not REFERENCE_LIMITS[0] <= value <= REFERENCE_LIMITS[1]:
        raise ValueError("釜温设定值应在 428–434 K")
    return float(value)


def neural_objective(z, model, state, previous, inlet, ref, move_weight):
    v = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    flows = NOMINAL_FLOW + FLOW_SCALE*v
    x = torch.as_tensor(state, dtype=torch.float64)
    d = torch.as_tensor(inlet, dtype=torch.float64)
    temps = []
    for u in flows.unbind():
        x = model(torch.cat((x, u.view(1), d)))
        temps.append(x[1])
    du = torch.diff(flows, prepend=flows.new_tensor([previous]))/FLOW_SCALE
    tracking = (torch.stack(temps)-ref)/TEMP_SCALE
    cost = tracking.square().sum() + move_weight*du.square().sum()
    grad, = torch.autograd.grad(cost, v)
    return cost.item(), grad.detach().numpy()


@njit(cache=True)
def mechanistic_objective(z, state, previous, inlet, ref, move_weight):
    n = len(z)
    A = np.empty((n, 3, 3))
    B = np.empty((n, 3))
    grad = np.zeros(n)
    error = np.empty(n)
    eps = np.array([1e-6, 1e-4, 1e-4])
    x = state.copy()
    cost = 0.0
    for k in range(n):
        flow = NOMINAL_FLOW + FLOW_SCALE*z[k]
        for j in range(3):
            shift = np.zeros(3)
            shift[j] = eps[j]
            A[k, :, j] = (integrate(x+shift, flow, inlet)-integrate(x-shift, flow, inlet))/(2*eps[j])
        B[k] = (integrate(x, flow+0.001, inlet)-integrate(x, flow-0.001, inlet))*(FLOW_SCALE/0.002)
        x = integrate(x, flow, inlet)
        error[k] = (x[1]-ref)/TEMP_SCALE
        delta = (flow-previous)/FLOW_SCALE
        cost += error[k]**2 + move_weight*delta**2
        grad[k] += 2*move_weight*delta
        if k:
            grad[k-1] -= 2*move_weight*delta
        previous = flow
    adj = np.zeros(3)
    for k in range(n-1, -1, -1):
        adj[1] += 2*error[k]/TEMP_SCALE
        grad[k] += np.dot(adj, B[k])
        adj = A[k].T @ adj
    return cost, grad


def solve(model, state, previous, ref, settings, warm=None, inlet=INLET, deadline=None):
    start = time.perf_counter()
    n = settings.horizon
    guess = np.full(n, previous) if warm is None else np.r_[warm[1:], warm[-1]]
    lo, hi = settings.flow_min, settings.flow_max

    def objective(z):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("MPC 求解超出控制周期预算")
        if model is None:
            return mechanistic_objective(z, state, previous, inlet, ref, settings.move_weight)
        return neural_objective(z, model, state, previous, inlet, ref, settings.move_weight)
    optimum = minimize(objective, (np.clip(guess, lo, hi)-NOMINAL_FLOW)/FLOW_SCALE,
                       method="SLSQP", jac=True,
                       bounds=[((lo-NOMINAL_FLOW)/FLOW_SCALE, (hi-NOMINAL_FLOW)/FLOW_SCALE)]*n,
                       options={"maxiter": 65, "ftol": 1e-7})
    seq = NOMINAL_FLOW + FLOW_SCALE*optimum.x
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("MPC 求解超出控制周期预算")
    if not optimum.success or not np.isfinite(optimum.fun) or not np.isfinite(seq).all():
        raise RuntimeError(f"MPC 求解失败：{optimum.message}")
    if min(seq) < lo-1e-6 or max(seq) > hi+1e-6:
        raise RuntimeError("MPC 流量约束校验失败")
    return np.clip(seq, lo, hi), {"iterations": int(optimum.nit), "cost": float(optimum.fun),
                                  "solve_ms": 1000*(time.perf_counter()-start)}
