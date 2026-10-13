"""The seven rplot.m figures grouped into four tabs, plus replay diagnostics."""
from matplotlib.figure import Figure
from .config import ATTACKS, PLOT_GROUPS
from ..control_plots import signal_legend

# Paper's recovery-time bound for the original six-attack experiment.
# This is a plot annotation, not a measured settling time or a solver parameter.
RECOVERY_TIME = 1.8627


def figures(data, *, duration=None):
    t = data["t"]
    horizon = duration if duration is not None else float(t[-1])
    result = {}
    for name, _ in PLOT_GROUPS:
        fig = Figure(figsize=(8.6, 6.2), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=.12, w_pad=.08)
        if name == "switching_replay":
            grid = fig.add_gridspec(2, 2)
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                    fig.add_subplot(grid[1, :])]
        else:
            axes = fig.subplots(2, 1)
        for ax in axes:
            ax.set_xlim(0, horizon)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.18)
            for start, end, _ in ATTACKS:
                if start < horizon:
                    ax.axvspan(start, min(end, horizon), color="#E69F00", alpha=.10, linewidth=0)
        top, bottom = axes[0], axes[-1]
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
            # Keep the negative attack transient visible (it falls below -0.5).
            bottom.set(ylabel="Estimation error", ylim=(-.6, .6))
            for index, (_, end, _) in enumerate(ATTACKS):
                if end >= horizon:
                    break
                next_start = ATTACKS[index + 1][0] if index + 1 < len(ATTACKS) else horizon
                bottom.axvspan(end, min(end + RECOVERY_TIME, next_start, horizon),
                               color="#228833", alpha=.12, linewidth=0,
                               gid=f"recovery-window-{index + 1}",
                               label="Recovery window" if index == 0 else None)
                if end + RECOVERY_TIME <= min(next_start, horizon):
                    bottom.axvline(end + RECOVERY_TIME, color="#228833", alpha=.7,
                                   linestyle=":", linewidth=1,
                                   gid=f"recovery-limit-{index + 1}",
                                   label=r"$T_r=1.8627$ s" if index == 0 else None)
        elif name == "state_control":
            top.plot(t, data["x2"], gid="live-x2", color="#0072BD", label=r"State $x_2$")
            top.set(ylabel=r"$x_2$", ylim=(-3, 3))
            bottom.plot(t, data["ut"], gid="live-ut", color="#0072BD", label=r"Control $u$")
            bottom.set(ylabel="Control input", ylim=(-20, 20))
        else:
            for ax, key, color in zip(axes[:2], ("K1", "K2"), ("#0072BD", "#D95319")):
                ax.plot(t, data[key], gid=f"live-{key}", color=color,
                        label=rf"$K(Z_{key[-1]})$")
                ax.set(ylabel="Switching function", ylim=(-.5, 1.5))
            bottom.plot(t, data["x1"], gid="live-x1", color="#0072BD", label=r"Live $x_1$")
            bottom.plot(t, data["xr1"], "--", gid="live-xr1", color="#D95319", label=r"Received $x_{r1}$")
            bottom.set(ylabel="Live / replayed measurement", ylim=(-2, 2))
        for ax in axes:
            signal_legend(ax)
        result[name] = fig
    return result
