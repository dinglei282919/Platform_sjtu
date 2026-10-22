"""Literal mathematical port of the eight Level-1 S-functions in PPTP.

State order: plant[5], rbf1[14], ndo1[2], beta2[1], ctrl[15].
The legacy RBF indexing, asymmetrical saturation, and mode endpoint rules
are preserved intentionally. There is no dependency on MATLAB at runtime.
"""
import math
import numpy as np

SWITCH_TIMES = np.array([.3, .65, 1.2, 1.7, 2.1, 2.8, 3.6, 4., 4.8,
                         5., 5.2, 5.8, 6.9, 7.4, 7.8, 9., 9.5])
CENTERS_1 = np.array([-4.5, -3., -1.5, 0., 1.5, 3., 4.5])
# ctrl.m uses d2(i), MATLAB column-major linear indexing, NOT d2(:, i).
CENTERS_2 = np.array([-4.5, -4.5, -3., -3., -1.5, -1.5, 0.])


def switching_signal(t):
    index = np.searchsorted(SWITCH_TIMES, t, side="right")
    # signal.m uses <= at the ends of these two intervals.
    if t == 2.1 or t == 2.8:
        index -= 1
    return int(1 + index % 2)


def initial_state():
    return np.r_[[-.5, 0., 0., 0., 0.], np.arange(1, 15) * .05,
                 [1., 5.], [0.], np.zeros(15)]


def performance(t, xi):
    eta = .9 * math.exp(3 * t / (t - 2)) + .1 if t < 2 else .1
    return eta, eta + .65 * (1 - eta) * math.tanh(xi)


def error_terms(z):
    if abs(z) > .1:
        return abs(z) - .1, math.copysign(1., z)
    return 0., 1 - 2 * math.cos(math.pi / 2 * math.sin(math.pi * (z + .1) / .4)**5)**5


class FdiModel:
    """Continuous dynamics plus history for the two Simulink Derivative blocks.

    Only accepted major steps may update history. Solver Newton iterations,
    rejected steps and finite-difference Jacobians must not mutate it.
    """

    def __init__(self):
        self.history = []

    def accept(self, t, eta1):
        self.history.append((float(t), float(eta1)))
        self.history = self.history[-2:]

    def signal_derivatives(self, t, eta1):
        for previous_t, previous_eta in reversed(self.history):
            if previous_t < t:
                dt = t - previous_t
                return ((math.sin(.75 * t) - math.sin(.75 * previous_t)) / dt,
                        (eta1 - previous_eta) / dt)
        # Both original Derivative blocks output zero at initialization.
        return 0., 0.

    def evaluate(self, t, state):
        x1, x2, w1, mu, xi = state[:5]
        theta1 = state[5:19].reshape(2, 7)
        gain, observer = state[19:21]
        beta = state[21]
        bound = state[22]
        theta2 = state[23:37].reshape(2, 7)
        sigma = switching_signal(t)
        yd = math.sin(.75 * t)
        eta, eta1 = performance(t, xi)
        dyd, deta = self.signal_derivatives(t, eta1)

        error = x1 - yd
        hx = error / (eta1 * math.sqrt(error * error + .1))
        z1 = hx / (1 - hx * hx)
        common = (1 + hx * hx) / (1 - hx * hx)**2
        k1 = common * .1 / (eta1 * (error * error + .1)**1.5)
        k2 = common * deta * error / (eta1**2 * math.sqrt(error * error + .1))
        nl1, sat1 = error_terms(z1)
        h1 = np.exp(-(x1 - CENTERS_1)**2 / 450)
        ef1 = theta1[sigma - 1] @ h1
        ew1 = observer + 10 * gain * x1
        aa = -(1 / k1 + 1.25 * k1 + .5 * k1 * gain**2) * sat1 * nl1
        ab = -1.1 * sat1
        ac = -k2 / k1
        alpha = aa + ab - ef1 - ew1 + dyd + ac

        z2 = x2 - beta - mu
        nl2, sat2 = error_terms(z2)
        h2 = np.exp(-((x1 - CENTERS_2)**2 + (x2 - CENTERS_2)**2) / 450)
        a = -3 * sat2 * nl2
        b = -h2 @ theta2[sigma - 1]
        c = -sat2 * bound
        ut = a + b + c - mu + (alpha - beta) / 1.1
        sut = float(np.clip(ut, -30, 35))
        fdi = x1 * x2 if sigma == 1 else x1**3 * math.sin(x2)
        velocity = (2 if sigma == 1 else 4 / 3) * x2
        gamma = -2 * x1 * math.cos(x1)**2 + .1 * x1 if sigma == 1 else -x1 * math.sin(x1)
        w2 = math.sin(t) if sigma == 1 else math.cos(t)
        limit = 35 if ut >= 0 else 30
        smooth_saturation = limit * math.tanh(ut / limit)

        dx = np.empty(37)
        # xi >= 0 is invariant for the model. A real extension of xi**.95
        # at negative solver trial states avoids complex Newton evaluations;
        # accepted states are separately checked against this invariant.
        dx[:5] = [x2 + w1, sut + fdi - velocity - .1 * velocity**3 - .01 * velocity**5 + w2,
                  -sigma * w1 + gamma, -mu + smooth_saturation - ut,
                  -xi - .1 * max(xi, 0.)**.95 + .12 * (max(ut - 35, 0.) + max(-30 - ut, 0.))]
        dtheta1 = -3000 * theta1
        dtheta1[sigma - 1] += k1 * sat1 * nl1 * h1
        dx[5:19] = dtheta1.ravel()
        d_gain = -.45 * gain**2 + .56 * gain * (1 + x1**2)
        dx[19:21] = [d_gain, -(sigma + 10 * gain) * ew1 - 10 * gain * (x2 + ef1) - 10 * x1 * d_gain]
        dx[21] = (alpha - beta) / 1.1
        dx[22] = -bound + nl2
        dtheta2 = -20 * theta2
        dtheta2[sigma - 1] += sat2 * nl2 * h2
        dx[23:37] = dtheta2.ravel()

        # The infinite initial bound is part of the original experiment.
        df = math.sqrt(.1) * eta1 / math.sqrt(1 - eta1**2) if eta1 < 1 else np.inf
        df1 = math.sqrt(.1) * eta / math.sqrt(1 - eta**2) if eta < 1 else np.inf
        return dx, dict(t=t, x1=x1, x2=x2, w1=w1, mu=mu, eta1=eta1, DF=df, DF1=df1,
                        L1=gain, ew1=ew1, yd=yd, dyd=dyd, d_eta=deta, k1=k1, k2=k2,
                        Nl1=nl1, satl1=sat1, ef1=ef1, alfa1=alpha, beta2=beta,
                        ut=ut, sut=sut, sig=sigma, fdi=fdi, e=error, ew_error=w1-ew1,
                        a=a, b=b, c=c, aa=aa, ab=ab, ac=ac)

    def rhs(self, t, state):
        return self.evaluate(t, state)[0]
