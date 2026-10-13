"""Four landscape figure pairs with the signals from plot0.m; no GUI dependency."""
from matplotlib.figure import Figure
import numpy as np
from ..control_plots import signal_legend


def figures(data, *, duration=None):
    t = data["t"]
    blue, red, green = "#0072BD", "#D95319", "#228833"
    result = {}
    for group in ("tracking", "disturbance", "state_control", "gain_switching"):
        fig = Figure(figsize=(12, 4.6), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=.12, w_pad=.08)
        left, right = fig.subplots(1, 2)
        for ax in (left, right):
            ax.set_xlim(0, duration if duration is not None else float(t[-1]))
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.18)
            ax.tick_params(direction="in", top=True, right=True)
        if group == "tracking":
            left.plot(t, data["yd"], gid="live-yd", color=red, label=r"Reference $y_d$")
            left.plot(t, data["x1"], "--", gid="live-x1", color=blue, label=r"Output $y$")
            left.set(ylabel="Output", ylim=(-1.2, 1.2))
            # Do not change the mathematical infinity in the saved data.
            bound = np.where(np.isfinite(data["DF"]), data["DF"], np.nan)
            right.plot(t, bound, "--", gid="live-DF", color=red, label="Upper modified FTPF")
            right.plot(t, -bound, "-.", gid="live-negative-DF", color=green, label="Lower modified FTPF")
            right.plot(t, data["e"], gid="live-e", color=blue, label=r"Tracking error $e$")
            right.set(ylabel="Tracking error", ylim=(-.8, .8))
        elif group == "disturbance":
            left.plot(t, data["w1"], gid="live-w1", color=red, label=r"Disturbance $w_1$")
            left.plot(t, data["ew1"], "--", gid="live-ew1", color=blue, label=r"Estimate $\hat w_1$")
            left.set(ylabel="Disturbance", ylim=(-1, 1))
            right.plot(t, data["ew_error"], gid="live-ew_error", color=blue, label=r"Estimation error $w_1-\hat w_1$")
            right.set(ylabel="Estimation error", ylim=(-.1, .1))
        elif group == "state_control":
            left.plot(t, data["x2"], gid="live-x2", color=blue, label=r"State $x_2$")
            left.set(ylabel=r"$x_2$", ylim=(-3, 3))
            right.plot(t, data["ut"], gid="live-ut", color=blue, label=r"Control $u$")
            right.plot(t, data["sut"], "--", gid="live-sut", color=red, label=r"Saturated control $\Gamma(u)$")
            right.set(ylabel="Control", ylim=(-80, 160))
        else:
            left.plot(t, data["L1"], gid="live-L1", color=blue, label=r"Dynamic high gain $L_1$")
            left.set(ylabel=r"$L_1$", ylim=(1, 2.5))
            right.step(t, data["sig"], where="post", gid="live-sig", color=blue, label=r"Switching signal $\sigma(t)$")
            right.set(ylabel="Mode", ylim=(.5, 2.5), yticks=[1, 2])
        for ax in (left, right):
            signal_legend(ax)
        result[group] = fig
    return result
