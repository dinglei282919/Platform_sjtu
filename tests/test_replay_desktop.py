"""Exercise the desktop worker lifecycle with an actual Qt event loop."""
import os
import json
from threading import Event
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from desktop.risk_control.replay_page import ReplayWidget
from platform_core.risk_control.replay.runner import ARTIFACT_NAMES
from platform_core.risk_control.replay.simulation import SimulationCancelled


class ReplayDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output_patch = patch("platform_core.risk_control.replay.runner.DEFAULT_OUTPUT_DIR", self.root / "runs")
        self.output_patch.start()
        self.widget = ReplayWidget()

    def pump_until(self, condition, timeout=20):
        end = time.monotonic() + timeout
        while not condition() and time.monotonic() < end:
            self.app.processEvents()
            # Release the Python GIL so the worker can import/render plots.
            time.sleep(.01)
        self.assertTrue(condition(), "replay worker did not finish within timeout")

    def tearDown(self):
        if self.widget._thread is not None:
            self.widget.cancel()
            self.pump_until(lambda: self.widget._thread is None)
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()
        self.output_patch.stop()
        self.temporary.cleanup()

    def test_completed_run_survives_navigation_and_exports_all_artifacts(self):
        self.widget.duration.setValue(.1)
        self.widget.start()
        self.assertFalse(self.widget.run_button.isEnabled())
        self.widget.show()
        self.widget.hide()  # Menu navigation hides, rather than destroys, this page.
        self.pump_until(lambda: self.widget._thread is None)
        self.assertIn("仿真完成", self.widget.status.text())
        self.assertEqual(self.widget.progress.value(), 100)
        self.assertEqual(self.widget.task_status.property("state"), "succeeded")
        self.assertEqual(self.widget.task_status.percentage.text(), "100%")
        self.assertFalse(self.widget.metrics.isHidden())
        self.assertTrue(all(value.text() for value in self.widget.metrics.values.values()))
        self.assertTrue(all(view.renderer().isValid() for view in self.widget.views.values()))
        tracking = self.widget.views["tracking"]
        tracking.canvas.zoom_by(1.2)
        for index in range(self.widget.plot_tabs.count()):
            self.widget.plot_tabs.setCurrentIndex(index)
            self.app.processEvents()
        self.widget.plot_tabs.setCurrentIndex(0)
        self.assertAlmostEqual(tracking.canvas._zoom, 1.2)
        self.assertTrue(self.widget.run_button.isEnabled())
        self.widget.show()
        self.assertTrue(self.widget.export_button.isEnabled())
        with patch("desktop.risk_control.replay_page.QFileDialog.getExistingDirectory", return_value=str(self.root)):
            self.widget.export()
        destination, = self.root.glob("replay-*")
        self.assertEqual({p.name for p in destination.iterdir()}, ARTIFACT_NAMES)
        for name in ARTIFACT_NAMES:
            self.assertEqual((destination / name).read_bytes(), (self.widget._run_dir / name).read_bytes())

    def test_live_plot_is_visible_before_completion_and_preserves_zoom(self):
        from platform_core.risk_control.replay.runner import run as actual_run
        release = Event()

        def gated_run(*args, progress, **kwargs):
            def report(percent, message):
                progress(percent, message)
                state = json.loads((self.widget._run_dir / "progress.json").read_text(encoding="utf-8"))
                if state["image_revision"] and percent < 85 and not release.is_set():
                    if not release.wait(15):
                        raise RuntimeError("preview test gate timed out")
            return actual_run(*args, progress=report, **kwargs)

        self.widget.duration.setValue(1.)
        self.widget.show()
        try:
            with patch("platform_core.risk_control.replay.runner.run", side_effect=gated_run):
                self.widget.start()
                self.pump_until(lambda: all(view.renderer().isValid() for view in self.widget.views.values()))
                self.assertTrue(self.widget.has_running_worker())
                self.assertEqual(self.widget.task_status.property("state"), "running")
                self.assertTrue(self.widget.metrics.isHidden())
                self.assertFalse(self.widget.export_button.isEnabled())
                self.widget.views["tracking"].canvas.zoom_by(1.2)
                self.widget.plot_tabs.setCurrentIndex(1)
                self.widget.plot_tabs.setCurrentIndex(0)
                release.set()
                self.pump_until(lambda: self.widget._thread is None)
                self.assertEqual(self.widget.task_status.property("state"), "succeeded")
                self.assertAlmostEqual(self.widget.views["tracking"].canvas._zoom, 1.2)
                self.assertFalse(self.widget._plot_refresh.timer.isActive())
        finally:
            release.set()

    def test_cancellation_close_guard_and_thread_cleanup(self):
        def cancellable_run(*args, cancelled, progress, **kwargs):
            progress(20, "正在运行")
            end = time.monotonic() + 5
            while not cancelled() and time.monotonic() < end:
                time.sleep(.01)
            raise SimulationCancelled("仿真已取消。")

        with patch("platform_core.risk_control.replay.runner.run", side_effect=cancellable_run):
            self.widget.start()
            self.pump_until(lambda: self.widget.progress.value() == 20)
            original_thread = self.widget._thread
            self.widget.start()
            self.assertIs(self.widget._thread, original_thread)
            event = QCloseEvent()
            with patch("desktop.risk_control.replay_page.QMessageBox.warning") as warning:
                self.widget.closeEvent(event)
            self.assertFalse(event.isAccepted())
            warning.assert_called_once()
            self.widget.cancel()
            self.widget._on_progress(50, "晚到的进度消息")
            self.assertIn("正在取消", self.widget.status.text())
            self.pump_until(lambda: self.widget._thread is None)
        self.assertIn("已取消", self.widget.status.text())
        self.assertEqual(self.widget.task_status.property("state"), "cancelled")
        self.assertTrue(self.widget.run_button.isEnabled())
        self.assertFalse(self.widget.cancel_button.isEnabled())
        self.assertFalse(self.widget.export_button.isEnabled())

    def test_worker_failure_is_reported_and_allows_retry(self):
        self.widget.metrics.set_values({"tracking_rmse": .02, "disturbance_rmse": 0.})
        self.assertEqual([value.text() for value in self.widget.metrics.values.values()], ["0.02000", "0.000"])
        with patch("platform_core.risk_control.replay.runner.run", side_effect=RuntimeError("测试求解失败")):
            self.widget.start()
            self.assertTrue(self.widget.metrics.isHidden())
            self.assertTrue(all(not value.text() for value in self.widget.metrics.values.values()))
            self.pump_until(lambda: self.widget._thread is None)
        self.assertIn("执行失败：测试求解失败", self.widget.status.text())
        self.assertEqual(self.widget.task_status.property("state"), "failed")
        self.assertTrue(self.widget.run_button.isEnabled())
        self.assertTrue(self.widget.duration.isEnabled())
        self.assertFalse(self.widget.export_button.isEnabled())
        self.assertFalse(self.widget.has_running_worker())
        self.widget.duration.setValue(.1)
        self.widget.start()
        self.assertEqual(self.widget.task_status.property("state"), "running")
        self.assertEqual(self.widget.task_status.percentage.text(), "0%")
        self.pump_until(lambda: self.widget._thread is None)
        self.assertEqual(self.widget.task_status.property("state"), "succeeded")


if __name__ == "__main__":
    unittest.main()
