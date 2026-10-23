"""Risk-control parameter scrolling must never step focused values."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QScrollArea, QSlider, QTabBar, QWidget, QLabel
from desktop.risk_control.process_control_dnn_mpc import ProcessControlDnnMpcWidget, DnnMpcWorkflowWidget
from desktop.risk_control.dnn_mpc_plot_view import ControlTabBar, ControlTabWidget
from desktop.risk_control.fdi_page import FdiWidget
from desktop.risk_control.replay_page import ReplayWidget
from desktop.risk_control.self_healing_page import SelfHealingWidget
from desktop.risk_control.reactor_page import ReactorWidget


class ParameterWheelTests(unittest.TestCase):
    def test_self_healing_scenarios_scroll_with_both_columns(self):
        app = QApplication.instance() or QApplication([])
        widget = SelfHealingWidget()
        try:
            widget.resize(1300, 650)
            widget.show()
            for _ in range(8):
                app.processEvents()
            scroll = widget._page_scroll
            bar = scroll.verticalScrollBar()
            tabs = widget.scenario_tabs
            field = widget.fields['duration']
            output = widget.task_status
            self.assertEqual(widget.findChildren(QScrollArea), [scroll])
            self.assertGreater(bar.maximum(), tabs.height() + 40)
            top = [item.mapTo(scroll.viewport(), QPoint()).y() for item in (tabs, field, output)]
            bar.setValue(tabs.height() + 40)
            app.processEvents()
            for item, initial in zip((tabs, field, output), top):
                self.assertEqual(initial - item.mapTo(scroll.viewport(), QPoint()).y(), bar.value())
            self.assertLess(tabs.mapTo(scroll.viewport(), QPoint()).y() + tabs.height(), 0)
            scroll.ensureWidgetVisible(field)
            field.setFocus()
            before_value, before_scroll = field.value(), bar.value()
            point = field.rect().center()
            event = QWheelEvent(QPointF(point), QPointF(field.mapToGlobal(point)), QPoint(), QPoint(0, -120),
                                Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
            app.sendEvent(field, event)
            app.processEvents()
            self.assertEqual(field.value(), before_value)
            self.assertGreater(bar.value(), before_scroll)
            scroll.ensureWidgetVisible(widget.run_button)
            app.processEvents()
            self.assertTrue(scroll.viewport().rect().contains(widget.run_button.mapTo(scroll.viewport(), widget.run_button.rect().center())))
            widget.logs.append('展开日志后仍可整页滚动访问。')
            widget.logs.toggle.setChecked(True)
            for _ in range(8):
                app.processEvents()
            scroll.ensureWidgetVisible(widget.logs.text)
            app.processEvents()
            self.assertTrue(scroll.viewport().rect().contains(widget.logs.text.mapTo(scroll.viewport(), widget.logs.text.rect().center())))
            for width in (1000, 1450):
                widget.resize(width, 650)
                for _ in range(8):
                    app.processEvents()
                for label in widget.findChildren(QLabel):
                    if label.isVisible() and label.wordWrap():
                        self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
        finally:
            widget.close()

    def test_dnn_stage_tabs_scroll_with_parameters_and_results(self):
        app = QApplication.instance() or QApplication([])
        widget = DnnMpcWorkflowWidget()
        try:
            widget.resize(1300, 650)
            widget.show()
            for _ in range(4):
                app.processEvents()
            bar = widget.verticalScrollBar()
            tabs = widget.tabBar()
            config = widget.training_page._config_group
            self.assertGreater(bar.maximum(), tabs.height())
            tabs_top = tabs.mapTo(widget.viewport(), QPoint()).y()
            config_top = config.mapTo(widget.viewport(), QPoint()).y()
            bar.setValue(tabs.height() + 20)
            app.processEvents()
            self.assertLess(tabs.mapTo(widget.viewport(), QPoint()).y() + tabs.height(), 0)
            self.assertEqual(tabs_top - tabs.mapTo(widget.viewport(), QPoint()).y(), bar.value())
            self.assertEqual(config_top - config.mapTo(widget.viewport(), QPoint()).y(), bar.value())

            field = widget.training_page._epochs_input
            widget.ensureWidgetVisible(field)
            app.processEvents()
            field.setFocus()
            before_scroll, before_value = bar.value(), field.value()
            point = field.rect().center()
            event = QWheelEvent(QPointF(point), QPointF(field.mapToGlobal(point)), QPoint(), QPoint(0, 120),
                                Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
            app.sendEvent(field, event)
            self.assertEqual(field.value(), before_value)
            self.assertLess(bar.value(), before_scroll)

            run = widget.training_page._run_btn
            widget.ensureWidgetVisible(run)
            app.processEvents()
            self.assertTrue(widget.viewport().rect().contains(run.mapTo(widget.viewport(), run.rect().center())))
            bar.setValue(0)
            QTest.mouseClick(tabs, Qt.LeftButton, pos=tabs.tabRect(1).center())
            app.processEvents()
            self.assertEqual(widget.currentIndex(), 1)
            self.assertEqual(bar.value(), 0)
            self.assertTrue(widget.mpc_page.isVisible())
            widget.setCurrentIndex(0)
            self.assertEqual(widget.training_page._epochs_input.value(), before_value)
            for width in (1000, 1450):
                widget.resize(width, 650)
                for _ in range(8):
                    app.processEvents()
                for label in widget.training_page.findChildren(QLabel, 'parameterRangeHint'):
                    self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
            viewport_width = widget.viewport().width()
            widget.training_page._logs.append('结果日志展开后仍可通过同一页面滚动条访问。')
            widget.training_page._logs.toggle.setChecked(True)
            for _ in range(8):
                app.processEvents()
            self.assertEqual(widget.viewport().width(), viewport_width)
            widget.ensureWidgetVisible(widget.training_page._logs.text)
            app.processEvents()
            log_center = widget.training_page._logs.text.rect().center()
            self.assertTrue(widget.viewport().rect().contains(widget.training_page._logs.text.mapTo(widget.viewport(), log_center)))
        finally:
            widget.close()

    def test_all_risk_control_tabs_ignore_wheel(self):
        app = QApplication.instance() or QApplication([])
        for make in (DnnMpcWorkflowWidget, FdiWidget, ReplayWidget, SelfHealingWidget, ReactorWidget):
            widget = make()
            try:
                widget.resize(1200, 700)
                widget.show()
                app.processEvents()
                bars = widget.findChildren(QTabBar)
                self.assertTrue(bars)
                for bar in bars:
                    with self.subTest(page=type(widget).__name__, tab=bar.accessibleName()):
                        self.assertIsInstance(bar, ControlTabBar)
                        bar.setCurrentIndex(1)
                        for delta in (QPoint(0, 120), QPoint(0, -120), QPoint(120, 0), QPoint(-120, 0)):
                            point = bar.tabRect(1).center()
                            event = QWheelEvent(QPointF(point), QPointF(bar.mapToGlobal(point)),
                                                QPoint(), delta, Qt.NoButton, Qt.NoModifier,
                                                Qt.NoScrollPhase, False)
                            app.sendEvent(bar, event)
                            self.assertEqual(bar.currentIndex(), 1)
            finally:
                widget.close()

    def test_tabs_still_accept_click_and_keyboard(self):
        app = QApplication.instance() or QApplication([])
        widget = ControlTabWidget()
        try:
            for label in ('一', '二', '三'):
                widget.addTab(QWidget(), label)
            widget.show()
            app.processEvents()
            bar = widget.tabBar()
            QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(1).center())
            self.assertEqual(widget.currentIndex(), 1)
            QTest.keyClick(bar, Qt.Key_Right)
            self.assertEqual(widget.currentIndex(), 2)
        finally:
            widget.close()

    def test_all_pages_keep_focused_values_and_keyboard_editing(self):
        app = QApplication.instance() or QApplication([])
        for make in (lambda: ProcessControlDnnMpcWidget("training"),
                     lambda: ProcessControlDnnMpcWidget("mpc"), FdiWidget,
                     ReplayWidget, SelfHealingWidget, ReactorWidget):
            widget = make()
            try:
                widget.resize(1200, 700)
                widget.show()
                app.processEvents()
                fields = widget.findChildren(QAbstractSpinBox) + widget.findChildren(QComboBox) + widget.findChildren(QSlider)
                self.assertTrue(fields)
                for field in fields:
                    with self.subTest(page=type(widget).__name__, field=field):
                        for area in widget.findChildren(QScrollArea):
                            if area.isAncestorOf(field):
                                area.ensureWidgetVisible(field)
                        field.setFocus()
                        app.processEvents()
                        value = field.currentIndex() if isinstance(field, QComboBox) else field.value()
                        for delta in (120, -120):
                            event = QWheelEvent(QPointF(8, 8), QPointF(field.mapToGlobal(QPoint(8, 8))),
                                                QPoint(), QPoint(0, delta), Qt.NoButton,
                                                Qt.NoModifier, Qt.NoScrollPhase, False)
                            app.sendEvent(field, event)
                            self.assertEqual(field.currentIndex() if isinstance(field, QComboBox) else field.value(), value)
                        if field.isEnabled() and isinstance(field, QAbstractSpinBox) and value < field.maximum():
                            QTest.keyClick(field, Qt.Key_Up)
                            self.assertGreater(field.value(), value)
            finally:
                widget.close()
