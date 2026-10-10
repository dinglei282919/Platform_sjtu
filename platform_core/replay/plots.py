"""The seven rplot.m figures grouped into four tabs, plus replay diagnostics."""
from matplotlib.figure import Figure
from .config import ATTACKS, PLOT_GROUPS


def figures(data, *, duration=None):
    t = data["t"]
    horizon = duration if duration is not None else float(t[-1])
    result = {}
    for name, _ in PLOT_GROUPS:
        fig = Figure(figsize=(8.6, 6.2), constrained_layout=True)
        axes = fig.subplots(2, 1)
        for ax in axes:
            ax.set_xlim(0, horizon)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.18)
            for start, end, _ in ATTACKS:
                if start < horizon:
                    ax.axvspan(start, min(end, horizon), color="#E69F00", alpha=.10, linewidth=0)
        top, bottom = axes
        if name == "tracking":
            top.plot(t, data["yd"], gid="live-yd", color="#0072BD", label=r"Reference $y_d$")
            top.plot(t, data["x1"], "--", gid="live-x1", color="#D95319", label=r"Output $y$")
            top.set(ylabel="Output", ylim=(-2, 2))
            bottom.plot(t, data["e"], gid="live-e", color="#0072BD", label=r"Tracking error $e$")
            bottom.set(ylabel="Tracking error", ylim=(-.2, .2))
        elif name == "disturbance":
            top.plot(t, data["w1"], gid="live-w1", color="#0072BD", label=r"Disturbance $w_1$")
            top.plot(t, data["ew1"], "--", gid="live-ew1", color="#D95319", label=r"Estimate $\hat w_1$")
            top.set(ylabel="Disturbance", ylim=(-.6, .6))
            bottom.plot(t, data["ew_error"], gid="live-ew_error", color="#0072BD", label=r"Estimation error $\tilde w_1$")
            bottom.axhline(.015, color="#D95319", linestyle="-.", label="Prescribed bound")
            bottom.axhline(-.015, color="#D95319", linestyle="-.")
            bottom.set(ylabel="Estimation error", ylim=(-.5, .5))
        elif name == "state_control":
            top.plot(t, data["x2"], gid="live-x2", color="#0072BD", label=r"State $x_2$")
            top.set(ylabel=r"$x_2$", ylim=(-3, 3))
            bottom.plot(t, data["ut"], gid="live-ut", color="#0072BD", label=r"Control $u$")
            bottom.set(ylabel="Control input", ylim=(-20, 20))
        else:
            top.plot(t, data["K1"], gid="live-K1", color="#0072BD", label=r"$K(Z_1)$")
            top.plot(t, data["K2"], "--", gid="live-K2", color="#D95319", label=r"$K(Z_2)$")
            top.set(ylabel="Switching functions", ylim=(-.1, 1.1))
            bottom.plot(t, data["x1"], gid="live-x1", color="#0072BD", label=r"Live $x_1$")
            bottom.plot(t, data["xr1"], "--", gid="live-xr1", color="#D95319", label=r"Received $x_{r1}$")
            bottom.set(ylabel="Live / replayed measurement", ylim=(-2, 2))
        for ax in axes:
            ax.legend(loc="upper right", fontsize=9, framealpha=.92)
        result[name] = fig
    return result
