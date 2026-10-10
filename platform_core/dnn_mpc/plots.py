"""Scientific figures shared by CS/BS previews and vector exports."""
from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib import rc_context
from matplotlib.ticker import MaxNLocator


def training_figure(history):
    fig = Figure(figsize=(8.6, 3.5), constrained_layout=True)
    ax = fig.subplots()
    epoch = [h["epoch"] for h in history]
    ax.plot(epoch, [np.sqrt(h["train_loss"]) for h in history], color="#0072BD", label="Training", lw=1.4)
    ax.plot(epoch, [np.sqrt(h["val_loss"]) for h in history], color="#D95319", label="Validation", lw=1.4)
    ax.set(xlabel="Epoch", ylabel="RMSE (normalized)")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=11)
    return fig


def simulation_figure(result, view="dnn_mpc"):
    fig = Figure(figsize=(8.6, 6.5))
    if view == "dnn_mpc":
        # Preserve the overview's original five-panel layout.
        grid = fig.add_gridspec(3, 4)
        axes = [fig.add_subplot(grid[row, 2*col:2*col+2]) for row in range(2) for col in range(2)]
        axes.append(fig.add_subplot(grid[2, 1:3]))
        indices = range(5)
        fig.subplots_adjust(left=.11, right=.97, bottom=.10, top=.96, hspace=.65, wspace=1.05)
    elif view == "mpc_states":
        axes = fig.subplots(2, 2).flatten()
        indices = range(4)
        fig.subplots_adjust(left=.11, right=.97, bottom=.10, top=.96, hspace=.4, wspace=.4)
    elif view == "mpc_control":
        axes, indices = [fig.subplots()], [4]
        fig.subplots_adjust(left=.12, right=.97, bottom=.12, top=.95)
    else:
        raise ValueError(f"Unknown MPC plot view: {view}")
    labels = [r"$x_1$ (m)", r"$x_2$ (m/s)", r"$x_3$ (rad)", r"$x_4$ (rad/s)", r"$u$ (N)"]
    def values(run):
        x = run["states"].copy()
        x[:, 2] -= np.pi
        return np.column_stack((x, run["controls"]))
    curves = values(result)
    for i, ax in zip(indices, axes):
        ax.plot(result["control_time"] if i == 4 else result["time"], curves[:, i], color="#0072BD", lw=1.6,
                marker="o" if len(curves) == 1 else None, markersize=3, label="Result")
        ax.set_ylabel(labels[i], fontsize=13)
        # Keep the full time span visible while only completed steps are drawn.
        duration = float(result.get("actual_duration", result["time"][-1]))
        ax.set_xlim(0, duration)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(direction="in", top=True, right=True, labelsize=11)
        ax.grid(alpha=.18)
        if i >= 2:
            ax.set_xlabel("Time (s)", fontsize=12)
    return fig


def png_bytes(figure):
    out = BytesIO()
    figure.savefig(out, format="png", dpi=180)
    return out.getvalue()


def vector_bytes(figure, format="svg"):
    if format not in ("svg", "pdf"):
        raise ValueError("Vector format must be svg or pdf")
    out = BytesIO()
    with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        figure.savefig(out, format=format)
    return out.getvalue()
