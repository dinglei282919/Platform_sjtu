"""DNN-MPC numerical, API, file I/O and desktop plot regression tests."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import errno
import hashlib
import json
import os
import time
import unittest

import h5py
import numpy as np
from scipy.io import savemat
from fastapi.testclient import TestClient

# Use a headless Qt platform when this test module creates its application.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from dnn_mpc_plot_view import DnnMpcPlotView

from platform_core.dnn_mpc import config, runner
from platform_core.dnn_mpc.data import load_dataset
from platform_core.dnn_mpc.generation import generate_dataset, plant_step
from platform_core.dnn_mpc.model import deserialize
from platform_core.dnn_mpc.simulation import INITIAL, REFERENCE, Q, R, ArrayPredictor, cost_gradient, simulate, terminal_weight
from platform_core.dnn_mpc.plots import simulation_figure, vector_bytes
from web_backend.task_manager import TaskManager
import web_backend.app as web


class DnnMpcTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output_patch = patch.object(config, "DEFAULT_OUTPUT_DIR", self.root / "output")
        self.state_patch = patch.object(config, "LATEST_TRAINING_STATE_PATH", self.root / "output" / "latest_training_state.json")
        self.output_patch.start()
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.output_patch.stop()
        self.temp.cleanup()

    def training_config(self):
        x, y = load_dataset(config.DEFAULT_DATASET)
        path = self.root / "data.mat"
        savemat(path, {"X_data": x[:80], "Y_data": y[:80]})
        return config.training_defaults() | dict(dataset_path=str(path), epochs=2, hidden_layers="8,8")

    def test_data_formats_and_rejection(self):
        x, y = load_dataset(config.DEFAULT_DATASET)
        self.assertEqual(x.shape, (20000, 5))
        for transpose in (False, True):
            raw = BytesIO()
            savemat(raw, {"X_data": x[:25].T if transpose else x[:25], "Y_data": y[:25].T if transpose else y[:25]})
            actual_x, actual_y = load_dataset(raw.getvalue())
            np.testing.assert_array_equal(actual_x, x[:25])
            np.testing.assert_array_equal(actual_y, y[:25])
        path = self.root / "v73.mat"
        with h5py.File(path, "w", userblock_size=512) as f:
            f["X_data"], f["Y_data"] = x[:25].T, y[:25].T
        np.testing.assert_array_equal(load_dataset(path)[0], x[:25])
        raw = BytesIO()
        savemat(raw, {"X_data": np.zeros((30, 6)), "Y_data": np.zeros((30, 4))})
        with self.assertRaises(ValueError):
            load_dataset(raw.getvalue())

    def test_generator_matches_original_transitions_and_sampling(self):
        original_x, original_y = load_dataset(config.DEFAULT_DATASET)
        np.testing.assert_allclose(plant_step(original_x[:, :4], original_x[:, 4]), original_y,
                                   rtol=1e-13, atol=1e-13)
        x, y, metadata = generate_dataset(20000)
        again_x, again_y, _ = generate_dataset(20000)
        np.testing.assert_array_equal(x, again_x)
        np.testing.assert_array_equal(y, again_y)
        self.assertFalse(np.array_equal(x, generate_dataset(20000, seed=43)[0]))
        self.assertEqual(x.shape, (20000, 5))
        self.assertEqual(y.shape, (20000, 4))
        np.testing.assert_allclose(x[:, :4].mean(axis=0), [0, 0, np.pi, 0], atol=.015)
        np.testing.assert_allclose(x[:, :4].std(axis=0), [.5, .5, .3, .5], atol=.015)
        self.assertTrue(np.all(np.abs(x[:, 4]) <= 10))
        self.assertAlmostEqual(x[:, 4].std(), 20 / np.sqrt(12), delta=.1)
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["integrator"], "explicit Euler")

    def test_generated_dataset_api_export_and_reimport(self):
        manager = TaskManager(self.root / "tasks")
        try:
            with patch.object(web, "task_manager", manager), TestClient(web.app) as client:
                defaults = client.get("/api/training/defaults").json()
                self.assertEqual(defaults["dataset_path"], str(config.DEFAULT_DATASET))
                self.assertEqual(web.TrainingTaskRequest().dataset_path, defaults["dataset_path"])
                cfg = defaults | dict(dataset_path="  ", sample_count=120, epochs=2, hidden_layers="8,8")
                for invalid in (99, 100001, True, 100.5):
                    response = client.post("/api/training/tasks", json=cfg | {"sample_count": invalid})
                    self.assertEqual(response.status_code, 422)
                    with self.assertRaises(ValueError):
                        config.validate_config("training", cfg | {"sample_count": invalid})
                response = client.post("/api/training/tasks", json=cfg)
                self.assertEqual(response.status_code, 200, response.text)
                task_id = response.json()["task_id"]
                task = self.wait_task(client, task_id)
                self.assertEqual(task["status"], "succeeded", task)
                result = task["result"]["result"]
                self.assertEqual(result["sample_count"], 120)
                self.assertEqual(result["dataset"]["kind"], "generated")
                self.assertEqual(result["dataset"]["seed"], 42)
                exported = client.get(f"/api/dnn/training/tasks/{task_id}/files/training_dataset.mat")
                self.assertEqual(exported.status_code, 200)
                x, y = load_dataset(exported.content)
                expected_x, expected_y, _ = generate_dataset(120)
                np.testing.assert_array_equal(x, expected_x)
                np.testing.assert_array_equal(y, expected_y)
                self.assertEqual(hashlib.sha256(exported.content).hexdigest(), result["dataset"]["sha256"])
                model = deserialize(Path(cfg["model_path"]).read_bytes())
                self.assertEqual(model.metadata["dataset"], result["dataset"])
                imported_path = self.root / "reimport.mat"
                imported_path.write_bytes(exported.content)
                imported = runner.run("training", cfg | {"dataset_path": str(imported_path), "sample_count": 1000})
                self.assertEqual(imported["result"]["sample_count"], 120)
                self.assertEqual(imported["result"]["dataset"]["kind"], "file")
                self.assertEqual(Path(imported["outputs"]["dataset_mat"]).read_bytes(), exported.content)
                # An invalid supplied path must fail, never silently generate a replacement.
                with self.assertRaises(FileNotFoundError):
                    runner.run("training", cfg | {"dataset_path": str(self.root / "missing.mat")})
        finally:
            manager.shutdown()

    def test_training_roundtrip_gradient_and_dynamic_horizon(self):
        cfg = self.training_config()
        payload = runner.run("training", cfg)
        raw = Path(cfg["model_path"]).read_bytes()
        model = deserialize(raw)
        self.assertEqual(model.metadata["activation"], "tanh")
        self.assertEqual(model.metadata["split"], "random")
        self.assertEqual(model.metadata["split_sizes"], [56, 12, 12])
        x, _ = load_dataset(cfg["dataset_path"])
        train_indices = np.random.default_rng(42).permutation(len(x))[:56]
        np.testing.assert_array_equal(model.scaler["x_mean"], x[train_indices].mean(axis=0))
        array = ArrayPredictor(model)
        np.testing.assert_allclose(array.step(INITIAL, .3)[0], model.step(INITIAL, .3), atol=1e-13)
        u = np.array([.1, -.2, .3])
        terminal, _ = terminal_weight(array)
        value, gradient = cost_gradient(u, INITIAL, array, terminal)
        eps = 1e-5
        numeric = [(cost_gradient(u+e, INITIAL, array, terminal)[0]-cost_gradient(u-e, INITIAL, array, terminal)[0])/(2*eps) for e in np.eye(3)*eps]
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-6)
        state, expected = INITIAL.copy(), 0.
        for force in u:
            expected += (state-REFERENCE) @ Q @ (state-REFERENCE) + R*force**2
            state = model.step(state, force)
        expected += (state-REFERENCE) @ terminal @ (state-REFERENCE)
        self.assertAlmostEqual(value, expected, places=9)
        result = simulate(model, duration=.24, horizon=3)
        observed = []
        def observe(completed, total, state, control):
            observed.append((completed, total, state.copy(), control))
            state[:] = 999  # A display callback must not mutate the simulation.
        streamed = simulate(model, duration=.24, horizon=3, on_step=observe)
        np.testing.assert_array_equal(streamed["states"], result["states"])
        np.testing.assert_array_equal(streamed["controls"], result["controls"])
        np.testing.assert_array_equal([step[2] for step in observed], result["states"])
        self.assertEqual([(step[0], step[1]) for step in observed], [(k, 12) for k in range(1, 13)])
        self.assertEqual(result["sequences"].shape, (12, 3))
        np.testing.assert_allclose(result["time"]-result["control_time"], .02)
        figure = simulation_figure(result)
        self.assertEqual(figure.axes[0].get_xlim(), (0., .24))
        np.testing.assert_array_equal(figure.axes[2].lines[0].get_ydata(), result["states"][:, 2]-np.pi)
        positions = [ax.get_position() for ax in figure.axes]
        self.assertAlmostEqual(positions[0].y0, positions[1].y0)
        self.assertAlmostEqual(positions[2].y0, positions[3].y0)
        self.assertLess(positions[0].x1, positions[1].x0)
        self.assertLess(positions[2].y1, positions[0].y0)
        self.assertLess(positions[4].y1, positions[2].y0)
        for position, axis in zip(positions, figure.axes):
            self.assertAlmostEqual(position.width, positions[0].width)
            self.assertEqual(axis.get_xlim(), (0., .24))
            np.testing.assert_array_equal(axis.get_xticks(), figure.axes[0].get_xticks())
        self.assertAlmostEqual((positions[4].x0+positions[4].x1)/2,
                               (positions[0].x0+positions[1].x1)/2)
        preview = result | {key: result[key][:1] for key in ("time", "control_time", "states", "controls")}
        preview_figure = simulation_figure(preview)
        self.assertEqual(preview_figure.axes[0].get_xlim(), (0., .24))
        self.assertEqual(len(preview_figure.axes[0].lines[0].get_xdata()), 1)
        self.assertNotIn(b"<image", vector_bytes(figure))
        self.assertEqual(config.mpc_defaults()["model_path"], cfg["model_path"])
        self.assertEqual(payload["result"]["model_sha256"], hashlib.sha256(raw).hexdigest())

    def test_short_horizon_terminal_cost_matches_lqr_and_settles(self):
        # Controllable, open-loop unstable linear cart-pole; no trained file dependency.
        a = np.eye(4) + .02*np.array([[0, 1, 0, 0], [0, 0, -.981, 0],
                                      [0, 0, 0, 1], [0, 0, 21.582, 0]])
        b = np.array([[0.], [.02], [0.], [-.04]])
        predictor = SimpleNamespace(step=lambda state, force: (
            REFERENCE + a @ (state - REFERENCE) + b[:, 0]*force, np.column_stack((a, b))))
        p, diagnostics = terminal_weight(predictor)
        gain = np.linalg.solve(np.array([[R]]) + b.T @ p @ b, b.T @ p @ a)
        self.assertGreater(np.max(np.abs(np.linalg.eigvals(a))), 1.)
        self.assertLess(diagnostics["local_closed_loop_radius"], 1.)
        np.testing.assert_allclose(a.T @ p @ a - p - a.T @ p @ b @ gain + Q, 0., atol=1e-8)
        initial = REFERENCE + [.01, 0., .005, 0.]
        expected_control = float((-gain @ (initial - REFERENCE))[0])
        model = SimpleNamespace(metadata={"dt": .02})
        for horizon in (1, 5):
            with self.subTest(horizon=horizon), patch("platform_core.dnn_mpc.simulation.ArrayPredictor", return_value=predictor), \
                    patch("platform_core.dnn_mpc.simulation.INITIAL", initial):
                result = simulate(model, duration=10., horizon=horizon)
            # SLSQP is approximate; compare both the action and value with analytic LQR.
            self.assertAlmostEqual(result["controls"][0], expected_control, delta=1e-3)
            optimal_cost = (initial - REFERENCE) @ p @ (initial - REFERENCE)
            self.assertAlmostEqual(result["logs"][0]["cost"], optimal_cost, delta=1e-6)
            self.assertTrue(all(entry["success"] for entry in result["logs"]))
            self.assertLess(np.max(np.abs(result["states"][-100:] - REFERENCE)), 5e-4)

    def test_unstabilizable_model_is_rejected(self):
        predictor = SimpleNamespace(step=lambda state, force: (
            state.copy(), np.column_stack((np.eye(4)*1.1, np.zeros(4)))))
        model = SimpleNamespace(metadata={"dt": .02})
        with patch("platform_core.dnn_mpc.simulation.ArrayPredictor", return_value=predictor):
            with self.assertRaisesRegex(ValueError, "终端权重"):
                simulate(model, duration=.2, horizon=5)

    def wait_task(self, client, task_id):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            task = client.get(f"/api/tasks/{task_id}").json()
            if task["status"] in ("succeeded", "failed"):
                return task
            time.sleep(.03)
        self.fail("DNN-MPC task did not finish")

    def test_api_training_simulation_exports_and_failure(self):
        manager = TaskManager(self.root / "tasks")
        try:
            with patch.object(web, "task_manager", manager), TestClient(web.app) as client:
                cfg = self.training_config()
                first = client.post("/api/training/tasks", json=cfg)
                self.assertEqual(first.status_code, 200, first.text)
                training_id = first.json()["task_id"]
                trained = self.wait_task(client, training_id)
                self.assertEqual(trained["status"], "succeeded", trained)
                self.assertEqual(trained["progress"]["percent"], 100)
                image = client.get(f"/api/training/tasks/{training_id}/images/training_rmse.svg")
                self.assertEqual(image.headers["content-type"], "image/svg+xml")
                self.assertIn(b"<svg", image.content)
                model = client.get(f"/api/dnn/training/tasks/{training_id}/files/trained_model.npz")
                self.assertEqual(model.content, Path(cfg["model_path"]).read_bytes())
                mpc_cfg = client.get("/api/mpc/defaults").json() | dict(sim_time=.2, prediction_horizon=3)
                self.assertEqual(mpc_cfg["model_path"], cfg["model_path"])
                info = client.post("/api/mpc/model-info", json={"model_path": cfg["model_path"]}).json()
                self.assertEqual(info["sha256"], trained["result"]["result"]["model_sha256"])
                first_step, resume = Event(), Event()
                def pause_after_first_step(*args, on_step, **kwargs):
                    def observe(completed, total, state, control):
                        on_step(completed, total, state, control)
                        if completed == 1:
                            first_step.set()
                            if not resume.wait(20):
                                raise TimeoutError("Preview inspection did not resume MPC")
                    return simulate(*args, on_step=observe, **kwargs)
                with patch("platform_core.dnn_mpc.simulation.simulate", pause_after_first_step):
                    try:
                        created = client.post("/api/mpc/tasks", json=mpc_cfg).json()
                        mpc_id = created["task_id"]
                        self.assertTrue(first_step.wait(20))
                        live = client.get(f"/api/tasks/{mpc_id}").json()
                        self.assertEqual(live["status"], "running")
                        self.assertIsNone(live["result"])
                        self.assertGreater(live["progress"]["percent"], 0)
                        self.assertLess(live["progress"]["percent"], 100)
                        live_revision = live["progress"]["image_revision"]
                        self.assertIsNotNone(live_revision)
                        preview = client.get(f"/api/mpc/tasks/{mpc_id}/images/dnn_mpc.svg")
                        self.assertEqual(preview.status_code, 200)
                        self.assertEqual(preview.headers["content-type"], "image/svg+xml")
                        self.assertIn(b"<svg", preview.content)
                    finally:
                        resume.set()
                    simulated = self.wait_task(client, mpc_id)
                self.assertEqual(simulated["status"], "succeeded", simulated)
                self.assertEqual(simulated["result"]["result"]["metrics"]["completed_steps"], 10)
                self.assertNotEqual(trained["result"]["artifact_dir"], simulated["result"]["artifact_dir"])
                image = client.get(f"/api/mpc/tasks/{mpc_id}/images/dnn_mpc.svg")
                self.assertEqual(image.headers["content-type"], "image/svg+xml")
                self.assertGreater(simulated["progress"]["image_revision"], live_revision)
                self.assertNotEqual(image.content, preview.content)
                zipped = client.get(f"/api/dnn/mpc/tasks/{mpc_id}/files/dnn_mpc_results.zip")
                self.assertEqual(zipped.status_code, 200)
                with ZipFile(BytesIO(zipped.content)) as bundle:
                    meta = json.loads(bundle.read("run.json"))
                    self.assertEqual(meta["horizon"], 3)
                    self.assertEqual(meta["model_sha256"], trained["result"]["result"]["model_sha256"])
                    self.assertEqual(np.asarray(meta["P"]).shape, (4, 4))
                    self.assertEqual(meta["terminal_weight"]["method"], "local_discrete_riccati")
                    self.assertLess(meta["terminal_weight"]["local_closed_loop_radius"], 1.)
                    trajectory = np.genfromtxt(BytesIO(bundle.read("trajectory.csv")), delimiter=",", names=True)
                    errors = np.column_stack([trajectory[name] for name in ("x1", "x2", "x3", "x4")]) - REFERENCE
                    stage_errors = np.vstack((INITIAL - REFERENCE, errors[:-1]))
                    expected_cost = (np.sum(stage_errors**2*np.asarray(meta["Q"]))
                                     + meta["R"]*np.sum(trajectory["u"]**2)
                                     + errors[-1] @ np.asarray(meta["P"]) @ errors[-1])
                    self.assertAlmostEqual(meta["metrics"]["total_cost"], expected_cost, places=9)
                    self.assertIn("control_sequences.csv", bundle.namelist())
                self.assertEqual(client.get(f"/api/dnn/mpc/tasks/{mpc_id}/files/config.py").status_code, 404)
                broken = client.post("/api/mpc/tasks", json=mpc_cfg | {"model_path": str(self.root / "missing.npz")}).json()
                failed = self.wait_task(client, broken["task_id"])
                self.assertEqual(failed["status"], "failed")
                self.assertIn("模型不存在", failed["error"])
                self.assertEqual(client.get(f"/api/mpc/tasks/{broken['task_id']}/images/dnn_mpc.svg").status_code, 404)
                # A failed run must not replace the last successfully trained model.
                self.assertEqual(config.mpc_defaults()["model_path"], cfg["model_path"])
        finally:
            manager.shutdown()

    def test_execution_lock_released_after_failure(self):
        with runner.execution_lock():
            with self.assertRaises(RuntimeError):
                with runner.execution_lock():
                    pass
        with self.assertRaises(FileNotFoundError):
            runner.run("mpc", config.mpc_defaults())
        with runner.execution_lock():
            pass



def windows_error(code):
    error = OSError(errno.EACCES, "file access conflict")
    error.winerror = code
    return error


class DnnMpcFileIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "progress.json"
        self.path.write_bytes(b'{"percent": 10}')

    def tearDown(self):
        self.temp.cleanup()

    @unittest.skipUnless(os.name == "nt", "Windows rename sharing semantics")
    def test_open_reader_releases_and_writer_recovers(self):
        blocked = Event()
        observed, failures = [], []
        replace = os.replace

        def replace_observed(source, destination):
            try:
                return replace(source, destination)
            except OSError as error:
                observed.append(error.winerror)
                blocked.set()
                raise

        def write():
            try:
                runner.atomic_write(self.path, b'{"percent": 20}')
            except Exception as error:
                failures.append(error)

        with patch.object(runner.os, "replace", side_effect=replace_observed):
            with self.path.open("rb") as reader:
                writer = Thread(target=write)
                writer.start()
                try:
                    self.assertTrue(blocked.wait(5), "Did not exercise a real sharing conflict")
                    self.assertEqual(reader.read(), b'{"percent": 10}')
                finally:
                    reader.close()
                    writer.join(5)
        self.assertFalse(writer.is_alive())
        self.assertTrue(observed)
        self.assertTrue(all(code in (5, 32, 33) for code in observed))
        self.assertEqual(failures, [])
        self.assertEqual(self.path.read_bytes(), b'{"percent": 20}')
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_persistent_conflict_is_bounded_and_preserves_old_file(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                original = windows_error(code)
                with patch.object(runner.os, "replace", side_effect=original) as replace, \
                        patch.object(runner.time, "sleep") as sleep:
                    with self.assertRaises(OSError) as raised:
                        runner.atomic_write(self.path, b"new")
                self.assertIs(raised.exception, original)
                self.assertEqual(replace.call_count, 11)
                self.assertEqual(sleep.call_count, 10)
                self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1.6)
                self.assertEqual(self.path.read_bytes(), b'{"percent": 10}')
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_unrelated_io_error_is_not_retried(self):
        original = OSError(errno.ENOSPC, "disk full")
        with patch.object(runner.os, "replace", side_effect=original) as replace, \
                patch.object(runner.time, "sleep") as sleep:
            with self.assertRaises(OSError) as raised:
                runner.atomic_write(self.path, b"new")
        self.assertIs(raised.exception, original)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b'{"percent": 10}')
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_cleanup_error_does_not_mask_replace_error(self):
        original = windows_error(5)
        with patch.object(runner.os, "replace", side_effect=original), \
                patch.object(runner.time, "sleep"), \
                patch.object(Path, "unlink", side_effect=OSError("cleanup failed")), \
                self.assertLogs(runner.__name__, level="WARNING"):
            with self.assertRaises(OSError) as raised:
                runner.atomic_write(self.path, b"new")
        self.assertIs(raised.exception, original)
        self.assertEqual(self.path.read_bytes(), b'{"percent": 10}')

    def test_failure_reporting_preserves_training_exception(self):
        original = ValueError("training failure")
        write_json = runner.write_json
        logs = []

        def fail_status(path, value):
            if value.get("message", "").startswith("执行失败"):
                raise windows_error(5)
            return write_json(path, value)

        with patch.object(config, "DEFAULT_OUTPUT_DIR", self.root / "output"), \
                patch.object(runner, "_training", side_effect=original), \
                patch.object(runner, "write_json", side_effect=fail_status):
            with self.assertRaises(ValueError) as raised:
                runner.run("training", config.training_defaults(), logs.append)
        self.assertIs(raised.exception, original)
        self.assertTrue(any("无法写入失败进度" in message for message in logs))


class DnnPlotViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "plot.svg"
        self.path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="860" height="650"><rect width="860" height="650" fill="white"/><path d="M50 100 L700 500" stroke="blue"/></svg>')
        self.view = DnnMpcPlotView()
        self.view.resize(1000, 500)
        self.view.show()
        self.app.processEvents()
        self.view.load(self.path)
        self.canvas = self.view.canvas

    def tearDown(self):
        self.view.close()
        self.temp.cleanup()

    def test_wheel_anchor_and_drag(self):
        point = QPoint(600, 150)
        before = self.canvas.mapToScene(point)
        event = QWheelEvent(QPointF(point), QPointF(self.canvas.viewport().mapToGlobal(point)),
                            QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        self.app.sendEvent(self.canvas.viewport(), event)
        self.assertAlmostEqual(self.canvas._zoom, 1.2)
        self.assertLess((self.canvas.mapToScene(point)-before).manhattanLength(), 5)
        center = self.canvas.mapToScene(self.canvas.viewport().rect().center())
        QTest.mousePress(self.canvas.viewport(), Qt.LeftButton, pos=point)
        QTest.mouseMove(self.canvas.viewport(), point + QPoint(100, 50), delay=20)
        QTest.mouseRelease(self.canvas.viewport(), Qt.LeftButton, pos=point + QPoint(100, 50))
        self.assertGreater((self.canvas.mapToScene(self.canvas.viewport().rect().center())-center).manhattanLength(), 20)

    def test_live_frame_preserves_zoom_and_position(self):
        self.canvas.zoom_by(2.)
        self.canvas.centerOn(350, 220)
        transform = self.canvas.transform()
        center = self.canvas.mapToScene(self.canvas.viewport().rect().center())
        self.path.write_text(self.path.read_text().replace('stroke="blue"', 'stroke="red"'))
        self.view.load(self.path)
        self.assertEqual(self.canvas.transform(), transform)
        self.assertLess((self.canvas.mapToScene(self.canvas.viewport().rect().center())-center).manhattanLength(), 5)

    def test_width_fit_limits_and_new_run_reset(self):
        self.canvas.fit("width")
        self.assertGreater(self.canvas._zoom, 1.)
        rendered = self.canvas._item.boundingRect().width() * self.canvas.transform().m11()
        self.assertAlmostEqual(rendered, self.canvas.viewport().width()-16)
        self.canvas.zoom_by(100.)
        self.assertEqual(self.canvas._zoom, 8.)
        self.canvas.zoom_by(.0001)
        self.assertEqual(self.canvas._zoom, .5)
        self.view.clear()
        self.view.load(self.path)
        self.assertEqual(self.canvas._zoom, 1.)

if __name__ == "__main__":
    unittest.main()
