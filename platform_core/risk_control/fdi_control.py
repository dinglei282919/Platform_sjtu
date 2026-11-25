"""Validated CSTR feedback observers and sampled controllers. Time is in minutes."""

from dataclasses import dataclass
import math

import numpy as np
from numba import njit
import torch

from . import cstr
from .thermal import plant_interval
from .healing import PI_KI, PI_KP


DT = cstr.SAMPLE_MIN
T_REF = float(cstr.NOMINAL[1])
TC_REF = float(cstr.NOMINAL[2])
Q_REF = cstr.NOMINAL_FLOW
Q_MIN, Q_MAX = cstr.FLOW_LIMITS
TEMP_NOISE = 0.025
JACKET_NOISE = 0.018
FLOW_NOISE = 0.06


@njit(cache=True)
def jacket_ekf(x, cov, previous_flow_read, previous_inlet, jacket, propagate):
    """Only the jacket instrument corrects this witness. No primary T input."""
    if propagate:
        step = cstr.INTEGRATION_MIN
        jac = np.empty((3, 3))
        for j, eps in enumerate((1e-6, 1e-4, 1e-4)):
            dx = np.zeros(3)
            dx[j] = eps
            plus = cstr.integrate(x+dx, previous_flow_read, previous_inlet, DT, step)
            minus = cstr.integrate(x-dx, previous_flow_read, previous_inlet, DT, step)
            jac[:, j] = (plus-minus)/(2*eps)
        x = cstr.integrate(x, previous_flow_read, previous_inlet, DT, step)
        cov = jac @ cov @ jac.T + np.diag(np.array([0.00012, 0.028, 0.018])**2)
    gain = cov[:, 2].copy() / (cov[2, 2] + JACKET_NOISE**2)
    x = x + gain*(jacket-x[2])
    transform = np.eye(3)
    transform[:, 2] -= gain
    cov = transform @ cov @ transform.T + JACKET_NOISE**2*np.outer(gain, gain)
    return x, (cov + cov.T)/2


class TemperaturePI:
    def __init__(self):
        self.integral = Q_REF

    def command(self, feedback):
        error = feedback - T_REF
        request = self.integral + PI_KP*error
        if not ((request >= Q_MAX and error > 0) or (request <= Q_MIN and error < 0)):
            self.integral += PI_KI*DT*error
        return float(np.clip(request, Q_MIN, Q_MAX))


@dataclass(frozen=True)
class Gate:
    center: float = 0.00266900469034681
    enter: float = 0.13046877898858036
    release: float = 0.06523438949429018
    enter_samples: int = 2
    release_samples: int = 3
    return_rate: float = 0.25


class MeasurementLoop:
    def __init__(self, mode, gate, fixed_weight):
        self.mode, self.gate, self.fixed_weight = mode, gate, fixed_weight
        self.x = cstr.NOMINAL.copy()
        self.cov = np.diag(np.array([0.001, 0.13, 0.09])**2)
        self.pi = TemperaturePI()
        self.weight = 1.0
        self.active = False
        self.enter_count = self.release_count = 0

    def observe(self, temperature, jacket, previous_flow_read, previous_inlet):
        if not np.all(np.isfinite((temperature, jacket, previous_flow_read))):
            if self.mode == "raw" and np.isfinite(temperature):
                return temperature, np.nan, np.nan, 1.0
            raise ValueError("反馈依据不足，缺少可信测温或上一拍流量回读")
        propagate = previous_inlet is not None
        known_inlet = previous_inlet if propagate else cstr.INLET
        self.x, self.cov = jacket_ekf(self.x, self.cov, previous_flow_read,
                                       known_inlet, jacket, propagate)
        if self.mode != "raw" and (not np.isfinite(self.x).all() or
                                   not 0 <= self.cov[1, 1] <= 0.13**2):
            raise ValueError("反馈依据不足，釜温估计不确定度超出启动标定范围")
        witness = self.x[1]
        residual = temperature - witness
        if self.mode == "guard":
            distance = abs(residual - self.gate.center)
            self.enter_count = self.enter_count + 1 if distance > self.gate.enter else 0
            self.release_count = self.release_count + 1 if distance < self.gate.release else 0
            if self.enter_count >= self.gate.enter_samples:
                self.active = True
            if self.release_count >= self.gate.release_samples:
                self.active = False
            self.weight = 0.0 if self.active else min(1.0, self.weight+self.gate.return_rate)
        elif self.mode == "fixed":
            self.weight = self.fixed_weight
        feedback = self.weight*temperature + (1.0-self.weight)*witness
        return feedback, witness, residual, self.weight


@njit(cache=True)
def heat_rate(concentration, temperature, inlet_temperature):
    reaction = 7.2e10*math.exp(-1e4/temperature)*concentration
    return (100./150.)*(inlet_temperature-temperature) + 200.*reaction


@njit(cache=True)
def observer_rhs(y, fraction, temperatures, jackets, inlet, load_estimate):
    measured_temp = temperatures[0] + fraction*(temperatures[1]-temperatures[0])
    jacket = jackets[0] + fraction*(jackets[1]-jackets[0])
    # Concentration is inferred from the measured temperature and material balance.
    reaction = 7.2e10*math.exp(-1e4/measured_temp)*y[0]
    dc = (100./150.)*(inlet[0]-y[0])-reaction
    dt = heat_rate(y[0], y[1], inlet[1]) + (7./1.5)*(jacket-y[1]) + load_estimate
    return np.array([dc, dt])


@njit(cache=True)
def complete_heat_balance(y, temperatures, jackets, inlet, load_estimate):
    n = round(DT/cstr.INTEGRATION_MIN)
    h = DT/n
    y = y.copy()
    for j in range(n):
        k1 = observer_rhs(y, j/n, temperatures, jackets, inlet, load_estimate)
        k2 = observer_rhs(y+h*k1/2, (j+0.5)/n, temperatures, jackets, inlet, load_estimate)
        k3 = observer_rhs(y+h*k2/2, (j+0.5)/n, temperatures, jackets, inlet, load_estimate)
        k4 = observer_rhs(y+h*k3, (j+1)/n, temperatures, jackets, inlet, load_estimate)
        y += h*(k1 + 2*k2 + 2*k3 + k4)/6
    return y


@dataclass(frozen=True)
class LoopSettings:
    vessel_gain: float = 2.5
    filter_min: float = 0.15
    load_rate: float = 0.045
    rbf_rate: float = 0.30
    robust_cap: float = 0.06


RBF_CENTERS = np.vstack((np.zeros(3), np.eye(3), -np.eye(3)))


def rbf_features(temperature, jacket, previous_command):
    scaled = np.array([(temperature-T_REF)/1., (jacket-TC_REF)/1.5,
                       (previous_command-Q_REF)/8.])
    return np.exp(-0.5*np.sum((scaled-RBF_CENTERS)**2, axis=1))


class ExecutionLoop:
    """A sampled two-level controller. Its only plant feedback is (T_m, Tc_m)."""

    def __init__(self, model, arm, settings):
        self.model, self.settings = model, settings
        self.outer = arm in ("outer", "full")
        self.inner = arm in ("inner", "full")
        self.balance = np.array(cstr.NOMINAL[:2])
        self.load_estimate = 0.0
        self.weights = np.zeros(len(RBF_CENTERS))
        self.robust = 0.0
        self.target = TC_REF
        self.previous_cmd = Q_REF
        self.previous_observation = self.previous_inlet = None
        self.sent_prediction = self.sent_features = None
        self.large_residuals = 0

    def predict(self, temperature, jacket, inlet, flows):
        rows = np.empty((len(flows), 7))
        rows[:, :3] = (self.balance[0], temperature, jacket)
        rows[:, 3], rows[:, 4:] = flows, inlet
        return self.model(torch.from_numpy(rows)).numpy()

    def update_observers(self, temperature, jacket):
        if self.previous_observation is None:
            self.balance[1] = temperature
            return 0., 0.
        last_t, last_j = self.previous_observation
        predicted = complete_heat_balance(
            self.balance, (last_t, temperature), (last_j, jacket),
            self.previous_inlet, self.load_estimate)
        t_residual = temperature-predicted[1]
        self.large_residuals = self.large_residuals+1 if abs(t_residual) > 0.075 else 0
        rate = self.settings.load_rate*(1.6 if self.large_residuals >= 2 else 1.0)
        self.load_estimate = float(np.clip(self.load_estimate + rate*t_residual/DT, -2., 2.))
        predicted[1] += 0.65*t_residual
        self.balance = predicted
        observed_effect = jacket-self.sent_prediction
        rbf_error = observed_effect-np.dot(self.weights, self.sent_features)
        # Prevent outward adaptation at a known command limit. No actual flow is read.
        outward = ((self.previous_cmd >= Q_MAX and rbf_error > 0)
                   or (self.previous_cmd <= Q_MIN and rbf_error < 0))
        if not outward:
            phi = self.sent_features
            update = self.settings.rbf_rate*rbf_error*phi/(0.05+np.dot(phi, phi))
            self.weights = np.clip(0.998*self.weights+update, -0.8, 0.8)
            excess = max(abs(jacket-self.target)-0.07, 0.)
            self.robust = min(self.settings.robust_cap, 0.90*self.robust+0.035*excess)
        return t_residual, observed_effect

    def command(self, temperature, jacket, inlet):
        if not np.all(np.isfinite((temperature, jacket))):
            raise ValueError("反馈依据不足，需要可信的釜温与夹套出口测温")
        # Restrict the demo to the high-branch initial-state box of the frozen
        # predictor's training design. Use only the controller's own signals.
        if (abs(temperature-T_REF) > 5.5 or abs(jacket-TC_REF) > 6.0 or
                abs(self.balance[0]-cstr.NOMINAL[0]) > 0.022):
            raise ValueError("温度或浓度估计超出冻结预测器的高温工作域")
        temp_residual, jacket_residual = self.update_observers(temperature, jacket)
        w = self.load_estimate if self.outer else 0.
        virtual = temperature + (-self.settings.vessel_gain*(temperature-T_REF)
                                  - heat_rate(self.balance[0], temperature, inlet[1]) - w)/(7./1.5)
        target_next = self.target + (-math.expm1(-DT/self.settings.filter_min))*(virtual-self.target)
        q = self.previous_cmd
        lo, hi = max(Q_MIN, q-0.1), min(Q_MAX, q+0.1)
        nominal = self.predict(temperature, jacket, inlet, (q, lo, hi))[:, 2]
        slope = (nominal[2]-nominal[1])/(hi-lo)
        if not math.isfinite(slope) or slope >= -0.02:
            raise ValueError("冻结 DNN 的局部冷却斜率超出验证范围")
        phi = rbf_features(temperature, jacket, q)
        fhat = float(np.dot(self.weights, phi))
        robust_term = self.robust*float(np.clip((jacket-self.target)/0.07, -1., 1.))
        correction = -(fhat+robust_term)/slope if self.inner else 0.
        request = q+(target_next-nominal[0])/slope+correction
        command = float(np.clip(request, Q_MIN, Q_MAX))
        # Predict again at the command actually issued. Store *before* the next observation.
        sent_prediction = self.predict(temperature, jacket, inlet, (command,))[0]
        diagnostics = dict(load_estimate=self.load_estimate, load_used=w, rbf=fhat,
                           robust=robust_term, inner_delta_q=correction, target_Tc=target_next,
                           slope=slope, vessel_innovation=temp_residual, jacket_innovation=jacket_residual,
                           dnn_T=sent_prediction[1], dnn_Tc=sent_prediction[2], C_hat=self.balance[0],
                           prior_dnn_Tc=self.sent_prediction, request=request)
        self.previous_cmd, self.previous_inlet = command, inlet.copy()
        self.previous_observation = (temperature, jacket)
        self.sent_prediction, self.sent_features = sent_prediction[2], phi
        self.target = target_next
        return command, diagnostics
