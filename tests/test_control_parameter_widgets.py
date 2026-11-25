"""Exercise the native spin-button hit regions used by all control pages."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QStyle, QStyleOptionSpinBox, QVBoxLayout, QWidget

from desktop.risk_control.controls import (
    ControlDoubleSpinBox, ControlSpinBox, control_page_style,
)
from desktop.risk_control.layout import FieldGrid, RunningValue


class ControlParameterWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setQuitOnLastWindowClosed(False)

    def test_native_buttons_step_both_number_types_and_stop_at_bounds(self):
        for factory, limit, step in ((ControlSpinBox, 2, 1), (ControlDoubleSpinBox, .2, .05)):
            with self.subTest(widget=factory.__name__):
                panel = QWidget()
                panel.setObjectName("parameterTest")
                panel.setStyleSheet(control_page_style("parameterTest"))
                layout = QVBoxLayout(panel)
                box = factory()
                box.setRange(-limit, limit)
                box.setSingleStep(step)
                layout.addWidget(box)
                panel.resize(300, 80)
                panel.show()
                self.app.processEvents()
                try:
                    option = QStyleOptionSpinBox()
                    box.initStyleOption(option)
                    up = box.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, box)
                    down = box.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, box)
                    self.assertGreaterEqual(up.width(), 24)
                    self.assertGreaterEqual(down.width(), 24)
                    self.assertFalse(up.intersects(down))
                    self.assertFalse(up.intersects(box.lineEdit().geometry()))
                    self.assertFalse(down.intersects(box.lineEdit().geometry()))
                    box.setValue(0)
                    QTest.mouseClick(box, Qt.LeftButton, pos=up.center())
                    self.assertAlmostEqual(box.value(), step)
                    QTest.mouseClick(box, Qt.LeftButton, pos=down.center())
                    self.assertAlmostEqual(box.value(), 0)
                    box.setValue(limit)
                    QTest.mouseClick(box, Qt.LeftButton, pos=up.center())
                    self.assertAlmostEqual(box.value(), limit)
                    QTest.mouseClick(box, Qt.LeftButton, pos=down.center())
                    self.assertAlmostEqual(box.value(), limit - step)
                    box.setValue(-limit)
                    QTest.mouseClick(box, Qt.LeftButton, pos=down.center())
                    self.assertAlmostEqual(box.value(), -limit)
                    QTest.mouseClick(box, Qt.LeftButton, pos=up.center())
                    self.assertAlmostEqual(box.value(), -limit + step)
                finally:
                    panel.close()
                    self.app.processEvents()

    def test_two_line_limits_and_pending_value_fit_in_a_compressed_field(self):
        panel = QWidget()
        form = FieldGrid(QVBoxLayout(panel))
        box = ControlDoubleSpinBox()
        box.setDecimals(6)
        form.add_field("T 半宽 (K)", box)
        bounds = form.add_hint(box, "下限 431.25 K\n上限 431.75 K")
        running = RunningValue(form, box)
        running.sync(2.154413)
        cell = form.fields[box]
        panel.show()
        try:
            box.setValue(.25)
            for width in (180, 360):
                with self.subTest(width=width):
                    panel.resize(width, 120)
                    self.app.processEvents()
                    cell.resize(cell.width(), 1)
                    self.app.processEvents()
                    self.assertTrue(running.isVisible())
                    self.assertGreaterEqual(bounds.height(), bounds.heightForWidth(bounds.width()))
                    self.assertGreaterEqual(running.height(), running.heightForWidth(running.width()))
                    self.assertLess(box.geometry().bottom(), bounds.geometry().top())
                    self.assertLess(bounds.geometry().bottom(), running.geometry().top())
                    self.assertLess(running.geometry().bottom(), cell.height())
        finally:
            panel.close()

    def test_invalid_edits_restore_the_last_committed_value_on_enter_and_focus_loss(self):
        for factory, valid, drafts in (
            (ControlSpinBox, "20", ["1", "41", "4.2", "", "3x", "NaN"]),
            (ControlDoubleSpinBox, "20.25", ["1", "41.5", "4.234", "", "3x", "inf"]),
        ):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            box, other = factory(), QPushButton()
            box.setRange(3, 40)
            box.setValue(30)
            box.setAccessibleName("测试设定值")
            clicks = []
            other.clicked.connect(lambda: clicks.append(True))
            layout.addWidget(box)
            layout.addWidget(other)
            panel.show()
            try:
                self.app.processEvents()
                box.setFocus()
                box.selectAll()
                QTest.keyClicks(box, valid)
                QTest.keyClick(box, Qt.Key_Return)
                self.assertIsNone(self.app.activeModalWidget())
                for key in (Qt.Key_Return, Qt.Key_Tab, Qt.LeftButton):
                    for draft in drafts:
                        with self.subTest(factory=factory, key=key, draft=draft):
                            box.setFocus()
                            box.selectAll()
                            QTest.keyClick(box, Qt.Key_Backspace)
                            QTest.keyClicks(box, draft)
                            self.assertEqual(box.value(), float(valid))
                            self.assertIsNone(self.app.activeModalWidget())
                            if key == Qt.LeftButton:
                                point = other.mapTo(panel, other.rect().center())
                                QTest.mouseClick(panel.windowHandle(), Qt.LeftButton, pos=point)
                            else:
                                QTest.keyClick(box, key)
                            self.app.processEvents()
                            self.assertEqual(box.value(), float(valid))
                            self.assertEqual(box.cleanText(), valid)
                            dialog = self.app.activeModalWidget()
                            self.assertIsInstance(dialog, QMessageBox)
                            self.assertEqual(dialog.windowTitle(), "参数输入无效")
                            self.assertIn("测试设定值", dialog.text())
                            self.assertIn("允许范围", dialog.text())
                            self.assertIn(valid, dialog.text())
                            self.assertEqual(clicks, [])
                            QTest.mouseClick(dialog.button(QMessageBox.Ok), Qt.LeftButton)
                            self.app.processEvents()
                            self.assertIsNone(self.app.activeModalWidget())
                box.setFocus()
                box.selectAll()
                QTest.keyClicks(box, "40")
                QTest.keyClick(box, Qt.Key_Return)
                self.assertIsNone(self.app.activeModalWidget())
                self.assertEqual(box.value(), 40)
            finally:
                panel.close()

    def test_warning_on_each_running_control_page(self):
        from desktop.risk_control.workspace import RiskWorkspace
        from platform_core.risk_control import NAVIGATION
        from platform_core.risk_control.service import ControlService
        from platform_core.task_manager import TaskManager

        with TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = TaskManager(root / "tasks")
            workspace = RiskWorkspace(ControlService(tasks, root / "cstr"))
            workspace.resize(1100, 760)
            workspace.show()
            try:
                for module, side in ((1, 0), (2, 0), (3, 0), (3, 1), (4, 0)):
                    with self.subTest(module=module, side=side):
                        workspace.select_page(NAVIGATION[module])
                        if module == 1:
                            page, box = workspace, workspace.controls["reference"]
                        elif module == 2:
                            page = workspace.healing_page
                            box = page.fields["speed"]
                        elif module == 3:
                            workspace.fdi_page.setCurrentIndex(side)
                            page = workspace.fdi_page.widget(side)
                            box = page.speed
                        else:
                            page = workspace.replay_page
                            box = page.speed
                        self.app.processEvents()
                        run, value = page.session.run, box.value()
                        self.assertTrue(box.isVisible() and box.isEnabled())
                        box.setFocus()
                        box.selectAll()
                        QTest.keyClicks(box, "9999")
                        QTest.keyClick(box, Qt.Key_Return)
                        self.app.processEvents()
                        dialog = self.app.activeModalWidget()
                        self.assertIsInstance(dialog, QMessageBox)
                        self.assertIn(box.accessibleName(), dialog.text())
                        QTest.keyClick(dialog, Qt.Key_Escape)
                        self.app.processEvents()
                        self.assertTrue(box.hasFocus())
                        self.assertEqual(box.value(), value)
                        self.assertEqual(page.session.run, run)
            finally:
                workspace.shutdown()
                workspace.close()
                tasks.shutdown()


if __name__ == "__main__":
    unittest.main()
