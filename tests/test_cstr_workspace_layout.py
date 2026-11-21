"""Check the desktop states that mirror the two CSTR workflow pages."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from desktop.risk_control.workspace import RiskWorkspace


class CstrWorkspaceLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.task = {"status": "running", "progress": {"percent": 50, "history": [(1, .3, .4)]}}
        service = SimpleNamespace(models=lambda: [], tasks={"job": self.task}, train=lambda *_: "job")
        self.w = RiskWorkspace(service)
        self.addCleanup(self.w.close)
        self.w.timer.stop()
        self.w.setAttribute(Qt.WA_DontShowOnScreen)
        self.w.resize(1440, 900)
        self.w.workflow.setCurrentIndex(0)
        with QSignalBlocker(self.w.pages):
            self.w.pages.setCurrentIndex(1)
        self.w.show()
        QTest.qWait(20)

    def test_fields_and_cards_reflow_when_the_workspace_is_resized(self):
        w = self.w
        stage = w.workflow.currentWidget()
        left, right = stage.panels
        self.assertEqual(left.y(), right.y())
        self.assertGreater(right.width(), left.width()*2)
        self.assertEqual(w.training_grid.columns, 2)
        w.resize(1030, 760)
        QTest.qWait(20)
        self.assertEqual(left.y(), right.y())
        self.assertEqual(w.training_grid.columns, 1)
        w.resize(700, 640)
        QTest.qWait(20)
        self.assertGreaterEqual(right.y(), left.y()+left.height())
        self.assertEqual(left.width(), right.width())
        w.resize(1440, 900)
        QTest.qWait(20)
        self.assertEqual(left.y(), right.y())
        self.assertEqual(w.training_grid.columns, 2)

    def test_csv_toggle_hides_only_generated_trajectory_fields(self):
        w = self.w
        w._set_csv(Path("external.csv"))
        self.assertEqual(w.file_name.text(), "external.csv")
        self.assertEqual(w.submit_train.text(), "使用外部数据训练")
        self.assertTrue(w.clear_csv.isVisible())
        for name, field in w.train_fields.items():
            self.assertEqual(field.isVisible(), name not in ("trajectories", "length"))
        w.clear_csv.click()
        self.assertTrue(all(field.isVisible() for field in w.train_fields.values()))
        self.assertEqual(w.submit_train.text(), "生成轨迹并训练")

    def test_training_results_and_report_follow_the_current_task(self):
        w = self.w
        self.assertTrue(w.empty_loss.isVisible())
        self.assertFalse(w.loss.isVisible())
        self.assertFalse(w.export_train.isVisible())
        w.submit_train.click()
        w.refresh()
        self.assertTrue(w.loss.isVisible())
        self.assertFalse(w.empty_loss.isVisible())
        self.task.update(status="succeeded", result={"test": {"one_rmse": [.2]*3, "rollout_rmse": [.3]*3},
                                                     "model": None, "validation": {"reasons": ["验证误差过高"]}})
        w.refresh()
        self.assertTrue(w.export_train.isVisible())
        self.assertIn("验证误差过高", w.metrics.text())
        w.submit_train.click()
        self.assertFalse(w.metrics.isVisible())
        self.assertFalse(w.export_train.isVisible())
        self.assertTrue(w.empty_loss.isVisible())
        self.assertFalse(w.loss.isVisible())

