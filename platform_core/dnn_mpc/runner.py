"""Shared orchestration, artifacts and progress for CS and BS."""
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from threading import Lock
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import hashlib
import json
import logging
import os
import time
from . import config as settings

_PROCESS_LOCK = Lock()
ARTIFACTS = {
    "training": {"training_rmse.svg", "training_rmse.pdf", "trained_model.npz",
                 "training_history.csv", "training_metrics.json", "training_summary.json", "training_dataset.mat"},
    "mpc": {"dnn_mpc.svg", "dnn_mpc.pdf", "simulation.npz", "trajectory.csv",
            "control_sequences.csv", "simulation_metrics.json", "mpc_validation_summary.json",
            "dnn_mpc_results.zip", "run.json"},
}


def new_run_dir(config):
    return Path(config["output_dir"]) / "runs" / uuid4().hex


def atomic_write(path, data):
    """Publish complete files, allowing brief Windows reader/rename conflicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        # Python/Qt readers may temporarily omit FILE_SHARE_DELETE on Windows.
        # Keep the old file intact while retrying; never unlink it as a fallback.
        for attempt in range(11):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 10:
                    raise
                time.sleep(min(0.02 * 2 ** attempt, 0.2))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            # Cleanup failure must not hide the original write/replace exception.
            logging.getLogger(__name__).warning("无法删除临时文件 %s：%s", temporary, exc)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


@contextmanager
def execution_lock():
    """Coordinate CS/BS processes; the OS releases the lock after a crash."""
    if not _PROCESS_LOCK.acquire(blocking=False):
        raise RuntimeError("已有 DNN-MPC 任务运行中，请等待完成后重试。")
    handle, locked = None, False
    try:
        settings.DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        handle = (settings.DEFAULT_OUTPUT_DIR / ".execution.lock").open("a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("另一窗口中的 DNN-MPC 任务运行中，请等待完成后重试。") from exc
        yield
    finally:
        if handle is not None:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        _PROCESS_LOCK.release()


def artifact_path(mode, name, run_dir):
    if name not in ARTIFACTS[mode]:
        raise FileNotFoundError("结果文件不存在。")
    path = Path(run_dir) / name
    if not path.is_file():
        raise FileNotFoundError("结果尚未生成，请先运行对应模块。")
    return path


def latest_artifact_dir(mode):
    try:
        payload = json.loads((settings.DEFAULT_OUTPUT_DIR / f"latest_{mode}.json").read_text(encoding="utf-8"))
        return Path(payload["artifact_dir"])
    except (OSError, ValueError, KeyError, TypeError):
        return settings.DEFAULT_OUTPUT_DIR / "no_results"


def run(mode, config, log=lambda message: None, *, run_dir=None):
    config = settings.validate_config(mode, config)
    run_dir = Path(run_dir) if run_dir is not None else new_run_dir(config)
    with execution_lock():
        run_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        started = time.perf_counter()

        def progress(percent, message):
            image_path = outputs.get("training_figure" if mode == "training" else "simulation_figure")
            image_revision = Path(image_path).stat().st_mtime_ns if image_path else None
            write_json(run_dir / "progress.json", dict(percent=percent, message=message, module=mode,
                       timestamp=datetime.now(timezone.utc).isoformat(), outputs=dict(outputs), image_revision=image_revision))

        try:
            progress(0, ("读取训练数据" if config["dataset_path"] else "生成训练数据") if mode == "training" else "加载模型")
            operation = _training if mode == "training" else _simulation
            result = operation(config, run_dir, outputs, progress, log)
            summary = "training_summary.json" if mode == "training" else "mpc_validation_summary.json"
            outputs["summary_json"] = str(run_dir / summary)
            result.update(outputs=dict(outputs), elapsed_seconds=time.perf_counter() - started,
                          timestamp=datetime.now(timezone.utc).isoformat(), config=config,
                          execution_backend="python_dnn_mpc")
            write_json(run_dir / summary, result)
            payload = dict(mode=mode, backend="python_dnn_mpc", result=result,
                           result_json=json.dumps(result, ensure_ascii=False), outputs=dict(outputs),
                           json_path=outputs["summary_json"], model_path=result["model_path"],
                           artifact_dir=str(run_dir), image_revision=time.time_ns())
            if mode == "training":
                write_json(settings.LATEST_TRAINING_STATE_PATH, dict(format_version=1,
                    model_path=result["model_path"], model_sha256=result["model_sha256"],
                    output_dir=config["output_dir"], summary_json=outputs["summary_json"], timestamp=result["timestamp"]))
            write_json(settings.DEFAULT_OUTPUT_DIR / f"latest_{mode}.json", payload)
            progress(100, "执行完成")
            log("执行完成，结果已保存。")
            return payload
        except Exception as exc:
            try:
                progress(0, f"执行失败：{exc}")
            except OSError as progress_error:
                log(f"无法写入失败进度：{progress_error}")
            raise


def _training(config, directory, outputs, progress, log):
    import numpy as np
    import torch
    from .data import load_dataset
    from .train import train
    from .model import serialize
    from .plots import training_figure, vector_bytes
    if config["dataset_path"]:
        dataset = Path(config["dataset_path"])
        if not dataset.is_file():
            raise FileNotFoundError(f"训练数据不存在：{dataset}")
        dataset_raw = dataset.read_bytes()
        x, y = load_dataset(dataset_raw)
        dataset_info = dict(kind="file", source_path=str(dataset), sample_count=len(x))
        provenance = "Python training from supplied MAT state-space transitions"
        log(f"读取 {len(x)} 条状态转移；使用完整外部数据集，不按生成样本数截取。")
    else:
        from .generation import generate_dataset, dataset_bytes
        x, y, dataset_info = generate_dataset(config["sample_count"])
        dataset_raw = dataset_bytes(x, y, dataset_info)
        provenance = "Python training from generated original cart_pole_dynamics transitions"
        log(f"按原状态方程生成 {len(x)} 条状态转移；seed={dataset_info['seed']}，dt=0.02 s。")
    dataset_info["sha256"] = hashlib.sha256(dataset_raw).hexdigest()
    atomic_write(directory / "training_dataset.mat", dataset_raw)
    outputs["dataset_mat"] = str(directory / "training_dataset.mat")
    progress(2, f"数据准备完成：{len(x)} 条状态转移")
    epochs = config["epochs"]
    log(f"tanh / Adam，训练轮数 {epochs}。")

    def callback(history):
        epoch = history[-1]["epoch"]
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            atomic_write(directory / "training_rmse.svg", vector_bytes(training_figure(history)))
            outputs["training_figure"] = str(directory / "training_rmse.svg")
        progress(round(2 + 93 * epoch / epochs, 1), f"训练 {epoch}/{epochs} 轮；验证 RMSE {history[-1]['val_rmse']:.6f}")

    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            predictor, history, metrics = train(x, y,
                hidden=tuple(map(int, config["hidden_layers"].split(","))), activation="tanh",
                epochs=epochs, lr=config["lr"], split="random", callback=callback,
                provenance=provenance)
    finally:
        torch.set_num_threads(threads)
    predictor.metadata["dataset"] = dataset_info
    raw = serialize(predictor)
    atomic_write(directory / "trained_model.npz", raw)
    atomic_write(config["model_path"], raw)
    history_csv = StringIO(newline="")
    writer = csv.DictWriter(history_csv, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    atomic_write(directory / "training_history.csv", history_csv.getvalue().encode("utf-8"))
    metrics["test_rmse"] = float(np.sqrt(metrics["test_mse"]))
    write_json(directory / "training_metrics.json", metrics)
    figure = training_figure(history)
    atomic_write(directory / "training_rmse.svg", vector_bytes(figure))
    atomic_write(directory / "training_rmse.pdf", vector_bytes(figure, "pdf"))
    outputs.update(model_path=config["model_path"], training_figure=str(directory / "training_rmse.svg"),
                   history_csv=str(directory / "training_history.csv"), metrics_json=str(directory / "training_metrics.json"))
    return dict(model_path=config["model_path"], model_sha256=hashlib.sha256(raw).hexdigest(),
                sample_count=len(x), dataset=dataset_info, metrics=metrics, best_epoch=metrics["best_epoch"],
                test_rmse=metrics["test_rmse"], metadata=predictor.metadata)


def _simulation(config, directory, outputs, progress, log):
    import numpy as np
    from .model import deserialize
    from .simulation import simulate, REFERENCE, INITIAL
    from .plots import simulation_figure, vector_bytes
    model_path = Path(config["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在，请先训练或选择 NPZ 模型：{model_path}")
    raw = model_path.read_bytes()
    model = deserialize(raw)
    log(f"模型内闭环：SLSQP + 局部 Riccati 终端权重，时长 {config['sim_time']} s，预测步长 {config['prediction_horizon']}。")
    preview_states, preview_controls = [], []
    last_preview = None

    def on_step(completed, total, state, control):
        nonlocal last_preview
        preview_states.append(state)
        preview_controls.append(control)
        if last_preview is None or completed == total or time.monotonic() - last_preview >= .5:
            preview = dict(time=(np.arange(completed)+1)*.02, control_time=np.arange(completed)*.02,
                           states=np.asarray(preview_states), controls=np.asarray(preview_controls),
                           actual_duration=total*.02)
            atomic_write(directory / "dnn_mpc.svg", vector_bytes(simulation_figure(preview)))
            outputs["simulation_figure"] = str(directory / "dnn_mpc.svg")
            last_preview = time.monotonic()
        progress(round(95*completed/total, 1),
                 f"仿真 {completed}/{total} 步；已计算 {completed*.02:.2f}/{total*.02:.2f} s")

    result = simulate(model, duration=config["sim_time"], horizon=config["prediction_horizon"], on_step=on_step)
    errors = result["states"] - REFERENCE
    controller = result["controller"]
    q, terminal = np.asarray(controller["Q"]), np.asarray(controller["P"])
    stage_errors = np.vstack((INITIAL - REFERENCE, errors[:-1]))
    failures = sum(not item["success"] for item in result["logs"])
    metrics = dict(tracking_mse=float(np.mean(errors**2)), tracking_rmse=float(np.sqrt(np.mean(errors**2))),
                   rmse_per_state=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                   max_abs_control=float(np.max(np.abs(result["controls"]))),
                   input_violations=int(np.count_nonzero(np.abs(result["controls"]) > 10 + 1e-9)),
                   total_cost=float(np.sum(stage_errors**2*q) + controller["R"]*np.sum(result["controls"]**2)
                                    + errors[-1] @ terminal @ errors[-1]),
                   solver_failures=failures, completed_steps=len(result["controls"]), actual_duration=result["actual_duration"])
    figure = simulation_figure(result)
    atomic_write(directory / "dnn_mpc.svg", vector_bytes(figure))
    atomic_write(directory / "dnn_mpc.pdf", vector_bytes(figure, "pdf"))
    arrays = BytesIO()
    np.savez_compressed(arrays, **{name: result[name] for name in ("time", "control_time", "states", "controls", "sequences")})
    atomic_write(directory / "simulation.npz", arrays.getvalue())
    csv_data = StringIO(newline="")
    writer = csv.writer(csv_data)
    writer.writerow(["control_time", "state_time", "x1", "x2", "x3", "x4", "u", "x3_deviation"])
    writer.writerows(np.column_stack((result["control_time"], result["time"], result["states"], result["controls"], errors[:, 2])))
    atomic_write(directory / "trajectory.csv", csv_data.getvalue().encode("utf-8"))
    sequences_csv = StringIO(newline="")
    writer = csv.writer(sequences_csv)
    writer.writerow(range(config["prediction_horizon"]))
    writer.writerows(result["sequences"])
    atomic_write(directory / "control_sequences.csv", sequences_csv.getvalue().encode("utf-8"))
    run_info = dict(source="python_slsqp", model_sha256=hashlib.sha256(raw).hexdigest(), logs=result["logs"],
                    dt=.02, horizon=config["prediction_horizon"], requested_duration=config["sim_time"],
                    actual_duration=result["actual_duration"], **controller,
                    input_bounds=[-10, 10], initial_state=INITIAL.tolist(), reference=REFERENCE.tolist(),
                    validation_mode="model_in_the_loop", metrics=metrics,
                    plot_state_definition="deviation from [0,0,pi,0]; plotted x3 = raw x3 - pi",
                    csv_state_definition="x1..x4 are original states; x3_deviation is the plotted x3")
    write_json(directory / "run.json", run_info)
    write_json(directory / "simulation_metrics.json", metrics)
    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for name in ("trajectory.csv", "control_sequences.csv", "run.json", "simulation_metrics.json", "dnn_mpc.svg", "dnn_mpc.pdf"):
            bundle.writestr(name, (directory / name).read_bytes())
    atomic_write(directory / "dnn_mpc_results.zip", archive.getvalue())
    outputs.update(simulation_figure=str(directory / "dnn_mpc.svg"), results_zip=str(directory / "dnn_mpc_results.zip"), model_path=str(model_path))
    warning = f"{failures} 步未达到优化停止条件，已保留有界解；请查看求解诊断。" if failures else ""
    if warning:
        log(warning)
    return dict(model_path=str(model_path), model_sha256=run_info["model_sha256"], metrics=metrics,
                tracking_mse=metrics["tracking_mse"], warning=warning, validation_mode="model_in_the_loop")
