import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
import unittest

from desktop.risk_control.reactor_page import ReactorWidget as ReactorWindow, create_application


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_application()
        cls.app.setQuitOnLastWindowClosed(False)
        cls.window = ReactorWindow()
        cls.window.show()
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.app.processEvents()

    def setUp(self):
        self.window.reset()
        self.app.processEvents()

    def tearDown(self):
        if self.window.worker:
            self.window.cancel_run()
            self.wait_until(lambda: self.window.worker is None)
        self.window.timer.stop()

    def wait_until(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertTrue(predicate())

    def test_diagram_and_narrow_layout(self):
        self.window.resize(900,700)
        self.app.processEvents()
        d = self.window.diagram
        self.assertTrue(d.animation.isValid())
        self.window.toggle_animation()
        self.assertFalse(d.animated)
        self.assertFalse(d.pixmap().isNull())
        self.assertLessEqual(d.pixmap().width(),d.width())
        self.assertLessEqual(d.pixmap().height(),d.height())

    def test_first_movie_frame_is_fitted_before_display(self):
        from desktop.risk_control.reactor_page import Diagram
        from PySide6.QtCore import QEventLoop, QTimer
        diagram = Diagram()
        try:
            diagram.resize(600, 400)
            diagram.set_animated(True)
            frame = diagram.animation.currentImage()
            self.assertFalse(frame.isNull())
            self.assertLessEqual(frame.width(), diagram.width() - 16)
            self.assertLessEqual(frame.height(), diagram.height() - 16)
        finally:
            diagram.animation.stop()
            diagram.close()
        window = ReactorWindow()
        try:
            window.resize(900, 700)
            window.show()
            self.app.processEvents()
            sizes = []
            loop = QEventLoop()
            timer = QTimer()
            timer.timeout.connect(lambda: sizes.append((window.diagram.width(), window.diagram.height())))
            timer.start(20)
            QTimer.singleShot(240, loop.quit)
            loop.exec()
            timer.stop()
            self.assertGreater(len(sizes), 2)
            self.assertEqual(len(set(sizes)), 1, "Animation frames must not resize the diagram container")
        finally:
            window.close()

    def test_short_run_replay_reset_and_rerun(self):
        w = self.window
        w.inputs["duration"].setValue(.3)
        w.inputs["prescribed_time"].setValue(.2)
        w.diagram.set_animated(True)
        w.start_run()
        self.assertFalse(w.diagram.animated)
        self.assertFalse(w.animation_button.isEnabled())
        self.assertFalse(w.run_button.isEnabled())
        self.assertFalse(w.inputs["x1"].isEnabled())
        self.wait_until(lambda: w.worker is None)
        self.assertIsNotNone(w.result)
        self.assertIn("用时", w.status.text())
        self.assertTrue(w.diagram.animated)
        self.assertTrue(w.animation_button.isEnabled())
        self.assertTrue(w.export_button.isEnabled())
        self.assertEqual(len(w.figure.axes),6)
        self.assertTrue(w.plot_view.renderer().isValid())
        w.slider.setValue(w.slider.maximum())
        self.assertIn("0.300",w.readout.text())
        w.toggle_playback()
        self.assertTrue(w.timer.isActive())
        w.toggle_playback()
        w.reset()
        self.assertEqual(w.inputs["duration"].value(),3)
        w.start_run()
        self.assertIsNone(w.result)
        self.assertFalse(w.plot_view.canvas._loaded)
        self.assertFalse(any(button.isEnabled() for button in w.plot_view._tool_buttons))
        self.assertFalse(w.export_button.isEnabled())
        w.cancel_run()
        self.wait_until(lambda: w.worker is None)
        self.assertIn("取消",w.status.text())
        self.assertTrue(w.run_button.isEnabled())

    def test_plot_wheel_drag_and_double_click(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtTest import QTest
        from platform_core.risk_control.reactor import Config, simulate
        w = self.window
        w.accept_result(simulate(Config(duration=.3, prescribed_time=.2)))
        w.tabs.setCurrentIndex(1)
        self.app.processEvents()
        canvas = w.plot_view.canvas
        self.assertTrue(w.plot_view.renderer().isValid())
        point = canvas.viewport().rect().center()
        event = QWheelEvent(QPointF(point), QPointF(canvas.viewport().mapToGlobal(point)),
                            QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        self.app.sendEvent(canvas.viewport(), event)
        self.assertAlmostEqual(canvas._zoom, 1.2)
        center = canvas.mapToScene(point)
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=point)
        QTest.mouseMove(canvas.viewport(), point + QPoint(60, 30), delay=20)
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=point + QPoint(60, 30))
        self.assertGreater((canvas.mapToScene(point) - center).manhattanLength(), 20)
        QTest.mouseDClick(canvas.viewport(), Qt.LeftButton, pos=point)
        self.assertAlmostEqual(canvas._zoom, 1.)
        self.assertEqual(canvas._mode, "fit")
        w.plot_view._tool_buttons[1].click()
        self.assertAlmostEqual(canvas._zoom, 1.2)
        w.plot_view._tool_buttons[3].click()
        rendered_width = canvas._item.boundingRect().width() * canvas.transform().m11()
        self.assertAlmostEqual(rendered_width, canvas.viewport().width() - 16)

    def test_worker_exception_recovers_controls(self):
        from unittest.mock import patch
        with patch("desktop.risk_control.reactor_page.simulate",side_effect=FloatingPointError("test overflow")):
            self.window.start_run()
            self.wait_until(lambda: self.window.worker is None)
        self.assertIn("test overflow",self.window.status.text())
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertTrue(self.window.run_button.isEnabled())

    def test_platform_pages_coexist_after_directory_merge(self):
        from unittest.mock import patch
        from PySide6.QtGui import QCloseEvent
        from main_interface import MainWindow
        window = MainWindow()
        window.show()
        try:
            window._on_submodule_clicked("风险管控优化决策", "化学反应器自适应控制仿真")
            reactor = window._reactor_content_widget
            self.assertTrue(reactor.isVisible())
            self.assertTrue(reactor.diagram.animation.isValid())
            window._on_submodule_clicked("风险管控优化决策", "多变量过程自愈控制仿真")
            healing = window._self_healing_content_widget
            self.assertTrue(healing.isVisible())
            self.assertFalse(reactor.isVisible())
            window._on_submodule_clicked("风险管控优化决策", "DNN/MPC 控制训练与优化")
            self.assertTrue(window._process_mpc_content_widget.isVisible())
            self.assertFalse(healing.isVisible())
            for page in (reactor, healing):
                with patch.object(page, "has_running_worker", return_value=True), patch("main_interface.QMessageBox.warning"):
                    event = QCloseEvent()
                    window.closeEvent(event)
                    self.assertFalse(event.isAccepted())
            window._on_submodule_clicked("风险管控优化决策", "化学反应器自适应控制仿真")
            self.assertIs(window._reactor_content_widget, reactor)
            self.assertFalse(healing.isVisible())
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
