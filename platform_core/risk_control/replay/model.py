"""The 34 continuous states and original S-function equations.

State order: plant(3), rbf(7), ew1(1), alfa1(9), eta1, var1,
ctrl(9), eta2, var2, ew2. All initial states are zero.
The replay channels feed rbf and ew1 only, following the actual model wiring.
"""
import numpy as np

INITIAL_STATE = np.zeros(34)
CENTERS = np.arange(-8., 9., 2.)


def reference(t):
    return (np.sin(.5*t) + .5*np.sin(t), .5*np.cos(.5*t) + .5*np.cos(t),
            -.25*np.sin(.5*t) - .5*np.sin(t))


def deadzone(z):
    if abs(z) <= .05:
        return 0., 1 - 2*np.cos(.5*np.pi*np.sin(.25*np.pi*(z+.05)/.05)**2)**2
    return abs(z)-.05, float(np.sign(z))


def switching(values):
    values = np.abs(values)
    if np.any(values >= 1.):
        return 0.
    middle = values[values > .5]
    return float(np.prod(np.cos(.5*np.pi*np.sin(.5*np.pi*(middle**2-.25)/1.5)**2)**2))


def evaluate(t, state, received, reference_values=None):
    x1, x2, d = state[:3]
    theta, beta, v1 = state[3:10], state[10], state[11:20]
    eta1, var1 = state[20:22]
    v2, eta2, var2, ew2 = state[22:31], state[31], state[32], state[33]
    xr1, xr2 = received
    yd, dyd, ddyd = reference(t) if reference_values is None else reference_values
    N1, sat1 = deadzone(x1-yd)
    rN1, rsat1 = deadzone(xr1-yd)
    K1 = switching(np.array([x1, yd, dyd]))
    # Preserve rbf.m's h1(1:i)=... assignment: the final center overwrites
    # all seven basis entries. Correcting it would change the baseline.
    h = np.exp(-(xr1-9.)**2/450.)
    ef = float(theta.sum()*h)
    ew1 = .25*(beta+xr1)
    h1 = np.exp(-np.sum((np.array([x1, yd, dyd])[:, None]-CENTERS)**2, axis=0)/450.)
    ps1 = x1*x1+dyd*dyd
    alfa1 = -100*N1*sat1-1.05*sat1-K1*(h1@v1+sat1*var1)-(1-K1)*sat1*eta1*ps1-ef-ew1
    features = np.r_[var1, eta1, dyd, ddyd, yd, x2, x1, theta, v1]
    K2 = switching(features)
    N2, sat2 = deadzone(x2-alfa1)
    h2 = np.exp(-np.sum((features[:, None]-CENTERS)**2, axis=0)/450.)
    ps2 = x2*x2+3*yd*yd
    ut = -20*N2*sat2-K2*(h2@v2+sat2*var2)-(1-K2)*sat2*eta2*ps2-sat2*ew2
    dx = np.empty(34)
    dx[:3] = (x2-x1+.25*d, ut-4.9*np.sin(2*x1)+.8*(x2-x1)+.1*np.sin(.5*t), -3*d-1.1*x1)
    dx[3:10] = rN1*rsat1*h-2*theta
    dx[10] = -3.25*beta-xr2-3.25*xr1+ef
    # In alfa1/ctrl.m, row h minus column x expands to a 9x9 matrix;
    # sys(i)=dv(i) then selects its first column (the -8 center).
    dx[11:20] = K1*N1*sat1*h1[0]-v1
    dx[20:22] = ((1-K1)*N1*ps1-eta1, K1*N1-var1)
    dx[22:31] = K2*N2*sat2*h2[0]-v2
    dx[31:] = ((1-K2)*N2*ps2-eta2, K2*N2-var2, N2-ew2)
    signals = dict(yd=yd, x1=x1, x2=x2, w1=.25*d, xr1=xr1, xr2=xr2, e=x1-yd,
                   ew1=ew1, ew_error=.25*d-ew1, ut=ut, ef=ef, N1=N1, sat1=sat1, K1=K1,
                   rN1=rN1, rsat1=rsat1, N2=N2, sat2=sat2, K2=K2, alfa1=alfa1,
                   eta1=eta1, var1=var1, ps1=ps1, eta2=eta2, var2=var2, ps2=ps2,
                   ew2=ew2, normth2=np.linalg.norm(v2), dyd=dyd, ddyd=ddyd)
    return dx, signals
