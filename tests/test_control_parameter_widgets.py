"""Exercise the native spin-button hit regions used by all control pages."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSpinBox, QVBoxLayout, QWidget

from desktop.risk_control.dnn_mpc_plot_view import (
    ControlDoubleSpinBox, ControlSpinBox, control_page_style,
)


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


if __name__ == "__main__":
    unittest.main()
