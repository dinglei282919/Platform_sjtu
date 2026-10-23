"""Actual Qt event-loop checks for the new module's worker and result widgets."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from platform_core.risk_control.self_healing.config import defaults
from platform_core.risk_control.self_healing.simulation import SimulationCancelled
from desktop.risk_control.self_healing_page import SelfHealingWidget


class SelfHealingDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.output_patch = patch("platform_core.risk_control.self_healing.runner.DEFAULT_OUTPUT_DIR", Path(self.temp.name))
        self.output_patch.start()
        self.widget = SelfHealingWidget()
        self.widget.resize(1450, 850)
        self.widget.show()

    def tearDown(self):
        if self.widget.has_running_worker():
            self.widget.cancel()
            self.pump_until(lambda: not self.widget.has_running_worker())
        self.widget.close()
        self.output_patch.stop()
        self.temp.cleanup()

    def pump_until(self, condition, timeout=15000):
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: loop.quit() if condition() else None)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        timer.start(); deadline.start(timeout)
        loop.exec()
        timer.stop(); deadline.stop()
        self.assertTrue(condition(), "Qt worker did not finish in time")

    def quick(self):
        for key, value in {"duration": 1., "fault_start": .3, "training_batches": 8, "training_hold": .2}.items():
            self.widget.fields[key].setValue(value)

    def test_presets_failure_retry_plots_and_export(self):
        tabs = self.widget.scenario_tabs
        self.assertEqual(tabs.count(), 3)
        for index, scenario in enumerate(("water", "cstr_htc", "cstr_catalyst")):
            QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(index).center())
            self.assertEqual(self.widget.config(), defaults(scenario))
        tabs.setCurrentIndex(0)
        self.quick()
        with patch("platform_core.risk_control.self_healing.runner.run", side_effect=RuntimeError("test failure")):
            self.widget.start()
            self.pump_until(lambda: self.widget._thread is None)
        self.assertEqual(self.widget.task_status.property("state"), "failed")
        self.assertTrue(self.widget.run_button.isEnabled())
        self.widget.start()
        self.pump_until(lambda: self.widget._thread is None)
        self.assertEqual(self.widget.task_status.property("state"), "succeeded")
        self.assertTrue(self.widget.export_button.isEnabled())
        self.assertTrue(all(view.renderer().isValid() for view in self.widget.views.values()))
        self.assertIn("SCSC", self.widget.metrics.text())
        self.widget.plot_tabs.setCurrentIndex(1)
        with patch("desktop.risk_control.self_healing_page.QFileDialog.getExistingDirectory", return_value=self.temp.name):
            self.widget.export()
        exports = list(Path(self.temp.name).glob("self_healing-*/result.json"))
        self.assertEqual(len(exports), 1)
        QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(0).center())
        self.assertIsNotNone(self.widget._result)
        self.assertEqual(self.widget.fields["duration"].value(), 1.)
        QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(1).center())
        self.assertEqual(self.widget.config(), defaults("cstr_htc"))
        self.assertIsNone(self.widget._result)
        self.assertFalse(self.widget.export_button.isEnabled())
        self.assertEqual(self.widget.metrics.text(), "")
        self.assertEqual(self.widget.task_status.property("state"), "idle")
        self.widget.fields["duration"].setValue(12.)
        QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(0).center())
        self.assertEqual(self.widget.fields["duration"].value(), 1.)
        self.assertIsNotNone(self.widget._result)
        self.assertEqual(self.widget.task_status.property("state"), "succeeded")
        self.assertTrue(self.widget.export_button.isEnabled())
        self.assertIn("SCSC", self.widget.metrics.text())
        self.assertTrue(all(view.renderer().isValid() for view in self.widget.views.values()))
        self.assertEqual(self.widget.plot_tabs.currentIndex(), 1)
        self.assertIn("任务完成", self.widget.logs.text.toPlainText())
        QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(1).center())
        self.assertEqual(self.widget.fields["duration"].value(), 12.)
        self.assertIsNone(self.widget._result)

    def test_validation_cancel_duplicate_start_and_close_guard(self):
        self.widget.fields["fault_start"].setValue(1000)
        self.widget.start()
        self.assertIsNone(self.widget._thread)
        self.assertFalse(list(Path(self.temp.name).rglob("self_healing_config.json")))
        self.widget._reset_config()
        def work(*args, progress, cancelled, **kwargs):
            progress(20, "working")
            deadline = time.monotonic() + 5
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(.01)
            raise SimulationCancelled("仿真已取消。")
        with patch("platform_core.risk_control.self_healing.runner.run", side_effect=work):
            self.widget.start()
            self.pump_until(lambda: self.widget.progress.value() == 20)
            tabs = self.widget.scenario_tabs
            self.assertFalse(tabs.isEnabled())
            QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(1).center())
            self.assertEqual(tabs.currentIndex(), 0)
            thread = self.widget._thread
            self.widget.start()
            self.assertIs(self.widget._thread, thread)
            event = QCloseEvent()
            with patch("desktop.risk_control.self_healing_page.QMessageBox.warning"):
                self.widget.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.widget.cancel()
            self.pump_until(lambda: self.widget._thread is None)
        self.assertEqual(self.widget.task_status.property("state"), "cancelled")
        self.assertTrue(self.widget.scenario_tabs.isEnabled())
        self.assertTrue(self.widget.run_button.isEnabled())
        self.assertFalse(self.widget.export_button.isEnabled())
