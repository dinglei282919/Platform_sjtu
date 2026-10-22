"""Result plots and portable exports; no GUI imports."""
import csv
import json
from pathlib import Path
import numpy as np
from .core import Config


def load_result(directory):
    """Read exports without pickle; validate shapes before presenting a saved run."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    Config(**summary["config"]).validate()
    keys = ("t", "x1", "x2", "observer1", "observer2", "error1", "error2",
            "u", "gamma", "k", "mode", "v1", "v2", "alpha1", "alpha2", "theta1_norm", "theta2_norm")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in keys}
        states = archive["states"].copy()
    t = data["t"]
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Saved time grid is invalid")
    if any(a.shape != t.shape or not np.isfinite(a).all() for a in data.values()):
        raise ValueError("Saved trajectory is invalid")
    if states.shape != (37,len(t)) or not np.isfinite(states).all():
        raise ValueError("Saved internal states are invalid")
    if abs(t[0]) > 1e-10 or abs(t[-1] - summary["config"]["duration"]) > 1e-10:
        raise ValueError("Saved trajectory is incomplete")
    metrics = summary["metrics"]
    if any(not isinstance(value,(int,float)) or not np.isfinite(value) for value in metrics.values()):
        raise ValueError("Saved metrics are invalid")
    return {"config": summary["config"], "data": data, "states": states, "metrics": metrics}


def populate_figure(figure, result):
    figure.clear()
    figure.set_layout_engine('constrained')
    axes = figure.subplots(2, 3)
    d = result["data"]
    panels = [(("x1", "observer1"), "State x1 / estimate"),
              (("x2", "observer2"), "State x2 / estimate"),
              (("error1", "error2"), "Observer errors"),
              (("u",), "Control input u"),
              (("mode",), "Switching mode"),
              (("theta1_norm", "theta2_norm"), "Active weight norms")]
    for ax, (keys, title) in zip(axes.flat, panels):
        for i, key in enumerate(keys):
            ax.plot(d["t"], d[key], linestyle="--" if i else "-", linewidth=1.2,
                    label=key, drawstyle="steps-post" if key == "mode" else "default")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)



def export_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    d = result["data"]
    with (directory / "trajectory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(d)
        writer.writerows(zip(*d.values()))
    np.savez_compressed(directory / "trajectory.npz", **d, states=result["states"])
    summary = {"config": result["config"], "metrics": result["metrics"],
               "model": "Connected MATLAB/Simulink R2023a blocks; literal d2(i) indexing",
               "limitations": ["No MATLAB trajectory parity established", "No physical units inferred",
                               "Diagram animation is illustrative, independent of numerical states",
                               "Peak control is sampled, not a continuous-time extremum"]}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(13, 7), dpi=150)
    FigureCanvasAgg(figure)
    populate_figure(figure, result)
    figure.savefig(directory / "results.png")
    return directory
