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
from PySide6.QtCore import QEventLoop, QPoint, QPointF, QTimer, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from dnn_mpc_plot_view import DnnMpcPlotView, format_rmse

from platform_core.dnn_mpc import config, runner
from platform_core.dnn_mpc.data import load_dataset
from platform_core.dnn_mpc.generation import generate_dataset, plant_step
from platform_core.dnn_mpc.model import deserialize
from platform_core.dnn_mpc.simulation import INITIAL, REFERENCE, Q, R, ArrayPredictor, cost_gradient, simulate, terminal_weight
from platform_core.dnn_mpc.plots import simulation_figure, vector_bytes
from web_backend.task_manager import TaskManager

# App import constructs a session-only manager. Keep test collection away from
# the live service's runtime directory without changing the shared TaskManager.
_web_runtime = TemporaryDirectory(prefix="dnn-mpc-api-tests-")
_manager_init = TaskManager.__init__
with patch.object(TaskManager, "__init__", lambda self, runtime_root=None:
                  _manager_init(self, runtime_root or Path(_web_runtime.name))):
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

    def test_module_output_defaults_and_dynamic_api_model_handoff(self):
        training = config.training_defaults()
        self.assertEqual(Path(training["output_dir"]), self.root / "output" / "training")
        self.assertEqual(Path(training["model_path"]), self.root / "output" / "trained_model.npz")
        external_model = self.root / "user_model.npz"
        external_model.write_bytes(b"model location fixture")
        config.LATEST_TRAINING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.LATEST_TRAINING_STATE_PATH.write_text(json.dumps({
            "format_version": 1, "model_path": str(external_model),
            "output_dir": str(self.root / "custom_training"),
        }))
        mpc = config.mpc_defaults()
        self.assertEqual(mpc["model_path"], str(external_model))
        self.assertEqual(Path(mpc["output_dir"]), self.root / "output" / "mpc")
        self.assertEqual(web.TrainingTaskRequest().output_dir, training["output_dir"])
        self.assertEqual(web.MpcTaskRequest().output_dir, mpc["output_dir"])
        self.assertEqual(web.MpcTaskRequest().model_path, str(external_model))
        custom = str(self.root / "explicit_mpc_destination")
        self.assertEqual(config.validate_config("mpc", {"output_dir": custom})["output_dir"], custom)

    def test_adaptive_modules_share_the_output_namespace(self):
        from platform_core.control_paths import CONTROL_OUTPUT_ROOT, module_output_dir
        from platform_core.fdi import runner as fdi_runner
        from platform_core.replay import runner as replay_runner
        for module, adapter in (("fdi", fdi_runner), ("replay", replay_runner)):
            self.assertEqual(adapter.DEFAULT_OUTPUT_DIR, CONTROL_OUTPUT_ROOT / module)
            directory = module_output_dir(module, root=self.root)
            with patch.object(adapter, "DEFAULT_OUTPUT_DIR", directory):
                first, second = adapter.new_run_dir(), adapter.new_run_dir()
            self.assertEqual(first.parent, directory / "runs")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_invalid_latest_state_falls_back_in_core_api_and_desktop(self):
        from process_control_dnn_mpc import ProcessControlDnnMpcWidget
        app = QApplication.instance() or QApplication([])
        state_path = config.LATEST_TRAINING_STATE_PATH
        state_path.parent.mkdir(parents=True, exist_ok=True)
        fallback = str(config.DEFAULT_OUTPUT_DIR / "trained_model.npz")
        existing = self.root / "existing.npz"
        existing.write_bytes(b"existing model must remain unchanged")
        invalid = [[], None, 42, "bad", True, {},
                   {"format_version": True, "model_path": str(existing)},
                   {"format_version": "1", "model_path": str(existing)},
                   {"format_version": 2, "model_path": str(existing)}]
        invalid += [{"format_version": 1, "model_path": value}
                    for value in (None, [], {}, 42, "", " ", "relative.npz", str(self.root), str(self.root / "missing.npz"))]
        manager = TaskManager(self.root / "tasks")
        with patch.object(web, "task_manager", manager), TestClient(web.app) as client:
            for raw in [json.dumps(value) for value in invalid] + ["{", ""]:
                with self.subTest(raw=raw):
                    state_path.write_text(raw, encoding="utf-8")
                    self.assertEqual(config.mpc_defaults()["model_path"], fallback)
                    response = client.get("/api/mpc/defaults")
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["model_path"], fallback)
                    self.assertEqual(web.MpcTaskRequest().model_path, fallback)
                    widget = ProcessControlDnnMpcWidget(page_mode="mpc")
                    try:
                        widget.show()
                        app.processEvents()
                        widget._use_latest_model()
                        self.assertEqual(widget._model_path_input.text(), fallback)
                    finally:
                        widget.close()
                    self.assertEqual(state_path.read_text(encoding="utf-8"), raw)
        self.assertEqual(existing.read_bytes(), b"existing model must remain unchanged")

    def test_late_training_failures_preserve_publication_and_previous_results(self):
        cfg = self.training_config() | dict(epochs=1)
        successful = runner.run("training", cfg)
        paths = [Path(cfg["model_path"]), config.LATEST_TRAINING_STATE_PATH,
                 config.DEFAULT_OUTPUT_DIR / "latest_training.json"]
        originals = {path: path.read_bytes() for path in paths}
        previous = {path: path.read_bytes() for path in Path(successful["artifact_dir"]).rglob("*") if path.is_file()}
        original_write = runner.atomic_write
        phases = ("training_history.csv", "training_rmse.pdf", "training_summary.json",
                  "model", "latest_training_state.json", "latest_training.json", "completion", "log")
        for phase in phases:
            with self.subTest(phase=phase):
                run_dir = self.root / ("failed-" + phase)
                def fail_write(path, raw):
                    path = Path(path)
                    fails = (path == paths[0] if phase == "model" else path.name == phase)
                    if phase == "completion" and path.name == "progress.json":
                        fails = json.loads(raw)["percent"] == 100
                    if fails:
                        raise OSError("injected " + phase)
                    return original_write(path, raw)
                def log(message):
                    if phase == "log" and message.startswith("执行完成"):
                        raise OSError("injected log")
                with patch.object(runner, "atomic_write", side_effect=fail_write):
                    with self.assertRaisesRegex(OSError, "injected"):
                        runner.run("training", cfg | dict(lr=.002), log, run_dir=run_dir)
                self.assertEqual({path: path.read_bytes() for path in paths}, originals)
                self.assertEqual({path: path.read_bytes() for path in previous}, previous)
                self.assertEqual(json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))["percent"], 0)
                self.assertFalse(list(self.root.rglob("*.bak")))
        retried = runner.run("training", cfg | dict(lr=.002))
        actual_hash = hashlib.sha256(paths[0].read_bytes()).hexdigest()
        self.assertNotEqual(paths[0].read_bytes(), originals[paths[0]])
        self.assertEqual(json.loads(paths[1].read_bytes())["model_sha256"], actual_hash)
        self.assertEqual(json.loads(paths[2].read_bytes())["result"]["model_sha256"], actual_hash)
        self.assertEqual(retried["result"]["model_sha256"], actual_hash)
        self.assertEqual({path: path.read_bytes() for path in previous}, previous)

    def test_first_publication_failure_leaves_no_model_or_success_indexes(self):
        cfg = self.training_config() | dict(epochs=1)
        original_write = runner.write_json
        def fail_index(path, value):
            if Path(path) == config.LATEST_TRAINING_STATE_PATH:
                raise OSError("injected index failure")
            return original_write(path, value)
        with patch.object(runner, "write_json", side_effect=fail_index):
            with self.assertRaisesRegex(OSError, "injected index"):
                runner.run("training", cfg)
        for path in (Path(cfg["model_path"]), config.LATEST_TRAINING_STATE_PATH,
                     config.DEFAULT_OUTPUT_DIR / "latest_training.json"):
            self.assertFalse(path.exists())
        self.assertFalse(list(self.root.rglob("*.bak")))

    def test_failed_rollback_retains_original_model_backup(self):
        model = self.root / "published.npz"
        model.write_bytes(b"original model")
        staged = self.root / "staged.npz"
        staged.write_bytes(b"replacement model")
        state = dict(model_path=str(model))
        replace = runner._replace_file
        def deny_restore(source, destination):
            if Path(source).suffix == ".bak":
                raise OSError("injected restore failure")
            return replace(source, destination)
        with patch.object(runner, "_replace_file", side_effect=deny_restore):
            with self.assertRaisesRegex(RuntimeError, "备份已保留"):
                with runner._training_publication(staged, state, {}):
                    raise OSError("injected final failure")
        backups = list(self.root.glob("published.npz.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"original model")

    def test_web_training_export_failure_preserves_model_and_allows_retry(self):
        cfg = self.training_config() | dict(epochs=1)
        runner.run("training", cfg)
        model = Path(cfg["model_path"])
        before = model.read_bytes()
        original_write = runner.atomic_write
        def fail_export(path, raw):
            if Path(path).name == "training_rmse.pdf":
                raise OSError("injected PDF failure")
            return original_write(path, raw)
        manager = TaskManager(self.root / "tasks")
        with patch.object(web, "task_manager", manager), TestClient(web.app) as client:
            with patch.object(runner, "atomic_write", side_effect=fail_export):
                response = client.post("/api/training/tasks", json=cfg | dict(lr=.002))
                self.assertEqual(response.status_code, 200, response.text)
                failed = self.wait_task(client, response.json()["task_id"])
            self.assertEqual(failed["status"], "failed")
            self.assertIn("injected PDF", failed["error"])
            self.assertEqual(model.read_bytes(), before)
            defaults = client.get("/api/mpc/defaults")
            self.assertEqual(defaults.status_code, 200)
            self.assertEqual(defaults.json()["model_path"], str(model))
            retried = client.post("/api/training/tasks", json=cfg | dict(lr=.002))
            completed = self.wait_task(client, retried.json()["task_id"])
            self.assertEqual(completed["status"], "succeeded", completed)
            self.assertEqual(completed["result"]["result"]["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())

    def test_desktop_training_export_failure_restores_controls_and_retries(self):
        from process_control_dnn_mpc import ProcessControlDnnMpcWidget
        app = QApplication.instance() or QApplication([])
        cfg = self.training_config() | dict(epochs=1)
        runner.run("training", cfg)
        model = Path(cfg["model_path"])
        widget = ProcessControlDnnMpcWidget(page_mode="training")
        widget._dataset_path_input.setText(cfg["dataset_path"])
        widget._epochs_input.setValue(1)
        widget._hidden_layers_input.setText(cfg["hidden_layers"])
        widget._lr_input.setValue(.002)
        widget.resize(1400, 850)
        widget.show()
        app.processEvents()
        original_write = runner.atomic_write
        def fail_export(path, raw):
            if Path(path).name == "training_rmse.pdf":
                raise OSError("injected desktop PDF failure")
            return original_write(path, raw)
        def wait_for_worker():
            # Run Qt's event loop as the application does, giving Python workers
            # time to execute between queued signals instead of polling qWait.
            loop = QEventLoop()
            poll = QTimer()
            poll.setInterval(10)
            poll.timeout.connect(lambda: loop.quit() if widget._thread is None else None)
            timeout = QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            poll.start()
            timeout.start(60000)
            try:
                if widget._thread is not None:
                    loop.exec()
            finally:
                poll.stop()
                timeout.stop()
            self.assertIsNone(widget._thread, "Desktop worker did not finish")
        try:
            # Reuse the same window: later attempts also exercise deferred Qt
            # cleanup, restored controls, and publication after an earlier retry.
            for attempt in range(3):
                before = model.read_bytes()
                protected = [config.LATEST_TRAINING_STATE_PATH,
                             config.DEFAULT_OUTPUT_DIR / "latest_training.json"]
                previous_indexes = {path: path.read_bytes() for path in protected}
                with self.subTest(attempt=attempt), \
                        patch.object(runner, "atomic_write", side_effect=fail_export), \
                        patch("process_control_dnn_mpc.QMessageBox.warning") as warning:
                    QTest.mouseClick(widget._run_btn, Qt.LeftButton)
                    wait_for_worker()
                    warning.assert_called_once()
                self.assertEqual(widget._task_status.property("state"), "failed")
                self.assertTrue(widget._run_btn.isEnabled())
                self.assertFalse(widget._progress_timer.isActive())
                self.assertEqual(model.read_bytes(), before)
                for path, contents in previous_indexes.items():
                    self.assertEqual(path.read_bytes(), contents)
                QTest.mouseClick(widget._run_btn, Qt.LeftButton)
                wait_for_worker()
                self.assertEqual(widget._task_status.property("state"), "succeeded")
                self.assertTrue(widget._run_btn.isEnabled())
                self.assertFalse(widget._progress_timer.isActive())
                self.assertTrue(widget._image.renderer().isValid())
                self.assertTrue(all(button.isEnabled() for button in widget._download_buttons))
                self.assertEqual(widget._payload["result"]["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
        finally:
            widget.close()

    def test_desktop_result_layout_stays_stable_across_status_and_size_changes(self):
        from process_control_dnn_mpc import ProcessControlDnnMpcWidget
        app = QApplication.instance() or QApplication([])

        def settle(widget):
            previous = None
            for _ in range(12):
                app.processEvents()
                geometry = (widget._result_content.size(), widget._result_scroll.viewport().size(),
                            widget._result_scroll.verticalScrollBar().maximum())
                if geometry == previous:
                    return
                previous = geometry
            self.fail("Result layout did not settle")

        for mode in ("training", "mpc"):
            with self.subTest(mode=mode):
                widget = ProcessControlDnnMpcWidget(page_mode=mode)
                try:
                    widget.show()
                    for width, height in ((1400, 850), (1000, 640), (1600, 1200)):
                        widget.resize(width, height)
                        widget._clear_result()
                        settle(widget)
                        viewport_width = widget._result_scroll.viewport().width()
                        for state in ("running", "failed", "succeeded"):
                            widget._task_status.set_status(state, "状态变化后仍可继续操作", 50)
                            widget._metrics_label.setText("训练 RMSE：0.01234；验证 RMSE：0.05678")
                            widget._result_path_label.setText(str(self.root / ("results_" * 12) / "result.json"))
                            widget._logs.append("验证折叠和展开日志时图表区域保持稳定。")
                            widget._logs.toggle.setChecked(state != "running")
                            settle(widget)
                            self.assertEqual(widget._result_scroll.viewport().width(), viewport_width)
                            self.assertEqual(widget._result_scroll.horizontalScrollBar().maximum(), 0)
                            widget._result_scroll.ensureWidgetVisible(widget._caption)
                            settle(widget)
                            viewport = widget._result_scroll.viewport()
                            top = widget._caption.mapTo(viewport, QPoint(0, 0)).y()
                            self.assertGreaterEqual(top, 0)
                            self.assertLessEqual(top + widget._caption.height(), viewport.height())
                finally:
                    widget.close()

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
        for view, indices in (("mpc_states", range(4)), ("mpc_control", [4])):
            split = simulation_figure(result, view=view)
            self.assertEqual(len(split.axes), len(indices))
            for axis, i in zip(split.axes, indices):
                np.testing.assert_array_equal(axis.lines[0].get_ydata(), figure.axes[i].lines[0].get_ydata())
                np.testing.assert_array_equal(axis.lines[0].get_xdata(), figure.axes[i].lines[0].get_xdata())
                self.assertEqual(axis.get_xlim(), (0., .24))
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
                        for name in ("mpc_states.svg", "mpc_control.svg"):
                            split_preview = client.get(f"/api/mpc/tasks/{mpc_id}/images/{name}")
                            self.assertEqual(split_preview.status_code, 200)
                            self.assertIn(b"<svg", split_preview.content)
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
                    for name in ("mpc_states.svg", "mpc_states.pdf", "mpc_control.svg", "mpc_control.pdf"):
                        self.assertIn(name, bundle.namelist())
                        self.assertEqual(client.get(f"/api/dnn/mpc/tasks/{mpc_id}/files/{name}").status_code, 200)
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
    def test_rmse_precision_matches_web_including_zeros_and_rounding_boundaries(self):
        # Number.toPrecision(4) results: trailing zeros, ties, carry and notation limits.
        examples = [(0., "0.000"), (-0., "0.000"), (.02, "0.02000"),
                    (.06784, "0.06784"), (.02994, "0.02994"), (1., "1.000"),
                    (1.0625, "1.063"), (9.9995, "9.999"), (9999.5, "1.000e+4"),
                    (1e-6, "0.000001000"), (1e-7, "1.000e-7"),
                    (9.9999e-7, "0.000001000")]
        for value, expected in examples:
            with self.subTest(value=value):
                self.assertEqual(format_rmse(value), expected)

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

    def test_empty_space_glyphs_render_without_qt_warnings_or_file_changes(self):
        from PySide6.QtCore import QByteArray, qInstallMessageHandler
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer

        def render(renderer):
            image = QImage(160, 100, QImage.Format_ARGB32)
            image.fill(0xffffffff)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            return image

        messages = []
        previous = qInstallMessageHandler(lambda kind, context, message: messages.append(message))
        try:
            for family in ("SimHei", "DejaVuSans"):
                for geometry in ("", ' d=""', " d=' '"):
                    with self.subTest(family=family, geometry=geometry):
                        raw = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                               f'xmlns:xlink="http://www.w3.org/1999/xlink" width="160" height="100">'
                               f'<defs><path id="{family}-20"{geometry} transform="scale(.015625)"/>'
                               f'<path id="visible" d="M 5 5 L 35 5 L 20 30 Z"/></defs>'
                               f'<use xlink:href="#visible" fill="blue"/>'
                               f'<use xlink:href="#{family}-20" x="50"/>'
                               f'<use xlink:href="#visible" x="80" fill="red"/></svg>').encode()
                        original = QSvgRenderer(QByteArray(raw))
                        expected = render(original)
                        self.path.write_bytes(raw)
                        messages.clear()
                        self.view.load(self.path)
                        actual = render(self.view.renderer())
                        self.assertEqual(messages, [])
                        self.assertEqual(actual, expected)
                        self.assertEqual(self.path.read_bytes(), raw)
        finally:
            qInstallMessageHandler(previous)

    def test_svg_compatibility_preserves_real_outlines_and_unrelated_invalid_paths(self):
        from dnn_mpc_plot_view import _svg_for_qt
        raw = (b'<svg xmlns="http://www.w3.org/2000/svg"><defs>'
               b'<path id="Custom-20" d="M 0 0 L 10 10"/>'
               b'<path id="invalid-outline" d=""/></defs></svg>')
        self.assertEqual(_svg_for_qt(raw), raw)

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
        self.assertFalse(self.canvas._item.isVisible(), "The previous result must disappear before a new run")
        self.assertFalse(any(button.isEnabled() for button in self.view._tool_buttons))
        self.assertEqual(self.view._zoom_label.text(), "100%")
        self.view.load(self.path)
        self.assertEqual(self.canvas._zoom, 1.)
        self.assertTrue(self.canvas._item.isVisible())
        self.assertTrue(all(button.isEnabled() for button in self.view._tool_buttons))

    def test_mpc_tabs_preserve_zoom_and_select_export(self):
        from process_control_dnn_mpc import ProcessControlDnnMpcWidget
        defaults = dict(output_dir=self.temp.name, model_path=str(self.path.with_suffix('.npz')),
                        sim_time=.2, prediction_horizon=3)
        with patch('process_control_dnn_mpc.mpc_defaults', return_value=defaults):
            widget = ProcessControlDnnMpcWidget(page_mode='mpc')
            try:
                widget.resize(1400, 850)
                widget.show()
                self.app.processEvents()
                widget._outputs = {key: str(self.path) for _, _, key in config.MPC_PLOT_GROUPS}
                widget._refresh_image()
                widget._image.canvas.zoom_by(1.2)
                for index, (stem, _, key) in enumerate(config.MPC_PLOT_GROUPS):
                    widget._plot_tabs.setCurrentIndex(index)
                    self.app.processEvents()
                    self.assertTrue(widget._images[key].isVisible())
                    self.assertEqual(widget._selected_plot_file('dnn_mpc.svg'), stem+'.svg')
                    self.assertEqual(widget._selected_plot_file('dnn_mpc.pdf'), stem+'.pdf')
                    self.assertEqual(widget._selected_plot_file('dnn_mpc_results.zip'), 'dnn_mpc_results.zip')
                widget._plot_tabs.setCurrentIndex(0)
                self.app.processEvents()
                self.assertAlmostEqual(widget._image.canvas._zoom, 1.2)
                widget._task_status.set_status('succeeded', '执行完成', 100)
                widget._clear_result()
                self.assertEqual(widget._task_status.property('state'), 'idle')
                self.assertEqual(widget._progress_bar.value(), 0)
                self.assertTrue(widget._image.isVisible())
                self.assertTrue(all(not image.canvas._item.isVisible() for image in widget._images.values()))
                self.assertTrue(all(not button.isEnabled() for image in widget._images.values()
                                    for button in image._tool_buttons))
            finally:
                widget.close()

if __name__ == "__main__":
    unittest.main()
