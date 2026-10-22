"""Persist machine-readable trajectories and reviewable plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)}")


def save_result(result, output_dir, plots=True, *, scenario_subdirectory=True):
    directory = Path(output_dir) / result["config"]["scenario"] if scenario_subdirectory else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {key: result[key] for key in ("config", "reference", "tolerance", "input_names", "output_names", "metrics", "diagnostics")}
    summary["events"] = {name: run["events"] for name, run in result["runs"].items()}
    summary["implementation"] = "Python mechanism simulation; not a numerical reproduction of the paper tables"
    summary["metric_notes"] = "MAPE actual denominator: PDF Eq34; reference denominator: MATLAB. ATAE is sum(t*abs(error)) without dt."
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value, allow_nan=False), encoding="utf-8")
    arrays = {"time": result["time"], "reference": result["reference"],
              "training_x": result["training"]["inputs"], "training_y": result["training"]["outputs"]}
    for name, run in result["runs"].items():
        for key in ("outputs", "inputs", "measured"):
            arrays[f"{name}_{key}"] = run[key]
        data = np.column_stack([result["time"], run["outputs"], run["inputs"]])
        np.savetxt(directory / f"{name}.csv", data, delimiter=",", comments="",
                   header=",".join(["time", *result["output_names"], *result["input_names"]]))
    np.savez_compressed(directory / "trajectories.npz", **arrays)
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "window", "output", "rmse", "mape_actual_percent", "mape_reference_percent", "atae"])
        for name, windows in result["metrics"].items():
            for window, metrics in windows.items():
                for j, output in enumerate(result["output_names"]):
                    writer.writerow([name, window, output, *[metrics[k][j] for k in ("rmse", "mape_actual_percent", "mape_reference_percent", "atae")]])
    if plots:
        plot_result(result, directory)
    return directory


def plot_result(result, directory):
    from platform_core.risk_control.control_plots import SimulationPlotPublisher
    from .plots import figures
    publisher = SimulationPlotPublisher(directory, figures, result["config"]["duration"], lambda: None)
    try:
        publisher.publish(result, final=True)
    finally:
        publisher.close()
