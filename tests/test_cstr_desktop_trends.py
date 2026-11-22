"""Exercise viewport gestures, archived samples and chart layout through Qt."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSignalBlocker, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from desktop.risk_control.workspace import RiskWorkspace
from platform_core.risk_control.cstr import SAMPLE_MIN, TREND_INTERVALS, TREND_MAX_INTERVALS


class DesktopTrendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.workspace = w = RiskWorkspace(SimpleNamespace(models=lambda: []))
        self.addCleanup(w.close)
        w.timer.stop()
        with QSignalBlocker(w.pages):
            w.pages.setCurrentIndex(1)
        w.resize(930, 530)
        w.setAttribute(Qt.WA_DontShowOnScreen)
        w.show()
        self.rows = [dict(seq=i, t=i*SAMPLE_MIN, T=432.5+i*.001, T_ref=433.,
                          Qc=150+i*.002, Qc_cmd=150.) for i in range(761)]
        for row in self.rows[130:140]:
            row.update(trial_id="pulse", disturbance_phase="active", T_hold=434., Qc_hold=149.)
        w.last_run, w.last_seq = "sample-run", self.rows[-1]["seq"]
        w.rows = self.rows[-(TREND_MAX_INTERVALS+1):]
        w.session = SimpleNamespace(sample_window=Mock(side_effect=lambda run, first, count:
                                                       self.rows[first:first+count]))
        w._draw_window()
        self.app.processEvents()

    def plot_position(self, chart, x=.5, y=.5):
        plot = chart.chart().plotArea()
        point = QPointF(plot.left()+x*plot.width(), plot.top()+y*plot.height())
        return chart.mapFromScene(chart.chart().mapToScene(point))

    def wheel(self, chart, angle, x=.5, y=.5, modifiers=Qt.NoModifier):
        pos = self.plot_position(chart, x, y)
        event = QWheelEvent(QPointF(pos), QPointF(chart.viewport().mapToGlobal(pos)), QPoint(),
                            QPoint(0, angle), Qt.NoButton, modifiers, Qt.NoScrollPhase, False)
        self.app.sendEvent(chart.viewport(), event)
        self.app.processEvents()

    def drag(self, chart, dx, dy):
        start = self.plot_position(chart)
        end = start + QPoint(dx, dy)
        view = chart.viewport()
        QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, start)
        QTest.mouseMove(view, end)
        QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, end)
        self.app.processEvents()

    def test_hover_keeps_one_sample_through_redraws_until_pointer_moves_or_leaves(self):
        w = self.workspace
        w._pin_window(115)
        chart = w.temp
        self.app.processEvents()

        def hover(fraction):
            viewport = chart.viewport()
            pos = self.plot_position(chart, fraction)
            event = QMouseEvent(QEvent.MouseMove, QPointF(pos), QPointF(viewport.mapToGlobal(pos)),
                                Qt.NoButton, Qt.NoButton, Qt.NoModifier)
            self.app.sendEvent(viewport, event)

        hover((135-115)/TREND_INTERVALS)
        self.assertTrue(chart.sample_tip.isVisible())
        selected = chart.sample_time
        text = chart.sample_tip.text()
        self.assertIn(f"仿真时间 {selected:.1f} min", text)
        self.assertIn("恒流对照　434 K", text)
        self.assertIn("设定值　433 K", text)
        for start in (125, 240, 400):
            w._pin_window(start)
            self.app.processEvents()
            self.assertEqual(chart.sample_time, selected)
            self.assertEqual(chart.sample_tip.text(), text)
            self.assertTrue(chart.sample_tip.isVisible())
        hover(.7)
        self.assertNotEqual(chart.sample_time, selected)
        self.assertNotIn("恒流对照", chart.sample_tip.text())
        self.app.sendEvent(chart.viewport(), QEvent(QEvent.Leave))
        self.assertTrue(chart.sample_tip.isHidden())

    def test_plain_wheel_scrolls_page_and_toolbar_keeps_time_window_synchronized(self):
        w = self.workspace
        self.assertGreaterEqual(w.flow.y(), w.temp.y()+w.temp.height())
        self.assertEqual(w.temp.width(), w.flow.width())
        self.assertGreaterEqual(w.temp.height(), 260)
        panel = next(area for area in w.findChildren(QScrollArea) if area.isAncestorOf(w.flow))
        self.assertGreater(panel.verticalScrollBar().maximum(), 0)
        panel.ensureWidgetVisible(w.temp)
        scroll = panel.verticalScrollBar()
        scroll.setValue(0)
        self.wheel(w.temp, -120)
        self.assertGreater(scroll.value(), 0)
        self.assertEqual(w.window_intervals, TREND_INTERVALS)
        self.assertIsNone(w.temp.manual_y_range)
        w.span_selector.setCurrentIndex(w.span_selector.findData(60))
        self.assertEqual(w.window_intervals, 60)
        self.assertIsNone(w.pinned)
        self.assertEqual(w.temp.domain, w.flow.domain)
        self.assertAlmostEqual(w.temp.domain[1], w.last_seq*SAMPLE_MIN)
        w._pin_window(115)
        self.assertTrue(w.temp.pulse_bands)
        w.span_selector.setCurrentIndex(w.span_selector.findData(30))
        self.assertEqual(w.window_intervals, 30)
        self.assertEqual(w.pinned, 130)
        w.session.sample_window.assert_called_with(w.last_run, w.pinned, 31)
        count = w.session.sample_window.call_count
        w._draw_window()
        self.assertEqual(w.session.sample_window.call_count, count)
        w.follow.click()
        w.span_selector.setCurrentIndex(w.span_selector.findData(TREND_MAX_INTERVALS))
        self.assertIsNone(w.pinned)
        self.assertEqual(w.temp.lines[0].count(), TREND_MAX_INTERVALS+1)
        self.assertEqual(w.session.sample_window.call_count, count)

    def test_y_zoom_stays_on_its_chart_and_reset_restores_automatic_ranges(self):
        w = self.workspace
        axis = w.temp.axes[1]
        before = axis.min(), axis.max()
        self.wheel(w.temp, 120, y=.25, modifiers=Qt.ShiftModifier)
        self.assertEqual(w.temp.y_zoom_level, 1)
        self.assertEqual(w.flow.y_zoom_level, 0)
        width = before[1]-before[0]
        self.assertAlmostEqual(axis.max()-axis.min(), width/2)
        self.assertAlmostEqual(axis.min()+.75*(width/2), before[0]+.75*width, delta=width*.01)
        manual = axis.min(), axis.max()
        w._pin_window(100)
        self.assertEqual((axis.min(), axis.max()), manual)
        w._reset_scales()
        self.assertIsNotNone(w.pinned)
        self.assertEqual(w.window_intervals, TREND_INTERVALS)
        self.assertEqual((axis.min(), axis.max()), w.temp.auto_y_range)

    def test_drag_both_axes_then_double_click_restores_live_and_both_y_ranges(self):
        w = self.workspace
        w.span_selector.setCurrentIndex(w.span_selector.findData(60))
        low, high = w.temp.auto_y_range
        self.drag(w.temp, 50, 25)
        self.assertIsNotNone(w.pinned)
        self.assertLess(w.pinned, w.last_seq-60)
        self.assertEqual(w.temp.domain, w.flow.domain)
        self.assertGreater(w.temp.manual_y_range[0], low)
        self.assertAlmostEqual(w.temp.manual_y_range[1]-w.temp.manual_y_range[0], high-low)
        self.assertIsNone(w.flow.manual_y_range)
        first = w.pinned
        self.drag(w.flow, 0, -16)
        self.assertEqual(w.pinned, first)
        self.assertIsNotNone(w.flow.manual_y_range)
        before = w.last_run, w.last_seq, list(w.rows)
        QTest.mouseDClick(w.flow.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(2, 2))
        self.assertEqual(w.pinned, first)
        QTest.mouseDClick(w.flow.viewport(), Qt.LeftButton, Qt.NoModifier, self.plot_position(w.flow))
        self.app.processEvents()
        self.assertIsNone(w.pinned)
        self.assertEqual(w.window_intervals, TREND_INTERVALS)
        self.assertEqual(w.temp.domain, w.flow.domain)
        self.assertAlmostEqual(w.temp.domain[1], w.last_seq*SAMPLE_MIN)
        for chart in (w.temp, w.flow):
            self.assertIsNone(chart.manual_y_range)
            self.assertEqual(chart.y_zoom_level, 0)
        self.assertEqual(before, (w.last_run, w.last_seq, w.rows))

    def test_vertical_drag_works_during_startup_and_retains_live_follow(self):
        w = self.workspace
        w.rows = self.rows[:5]
        w.last_seq = 4
        w._draw_window()
        self.drag(w.temp, 0, 24)
        manual = w.temp.manual_y_range
        self.assertIsNotNone(manual)
        self.assertIsNone(w.pinned)
        w.rows = self.rows[:TREND_INTERVALS+5]
        w.last_seq = w.rows[-1]["seq"]
        w._draw_window()
        self.assertEqual(w.temp.manual_y_range, manual)
        self.assertAlmostEqual(w.temp.domain[1], w.last_seq*SAMPLE_MIN)

    def test_startup_drag_into_blank_time_and_archived_edge_use_only_real_samples(self):
        w = self.workspace
        w.rows = self.rows[:5]
        w.last_seq = 4
        w._draw_window()
        self.assertEqual(w.temp.domain, (0, TREND_INTERVALS*SAMPLE_MIN))
        self.drag(w.temp, 60, 15)
        self.assertLess(w.pinned, 0)
        self.assertIsNotNone(w.temp.manual_y_range)
        self.assertIsNone(w.flow.manual_y_range)
        self.assertEqual(w.temp.domain, w.flow.domain)
        self.assertEqual(w.history_slider.value(), w.pinned)
        self.assertEqual(w.history_slider.minimum(), 1-TREND_INTERVALS)
        self.assertEqual(w.history_slider.maximum(), 3)
        self.assertEqual([p.x() for p in w.temp.lines[0].points()], [r["t"] for r in w.rows])
        w.session.sample_window.assert_not_called()
        w.history_slider.setValue(3)
        self.assertEqual([p.x() for p in w.temp.lines[0].points()], [r["t"] for r in w.rows[3:]])
        w.rows = self.rows[-(TREND_MAX_INTERVALS+1):]
        w.last_seq = self.rows[-1]["seq"]
        w.span_selector.setCurrentIndex(w.span_selector.findData(TREND_MAX_INTERVALS))
        w.history_slider.setValue(1-TREND_MAX_INTERVALS)
        w.session.sample_window.assert_called_with(w.last_run, 0, 2)
        self.assertEqual([p.x() for p in w.flow.lines[0].points()], [0., SAMPLE_MIN])
