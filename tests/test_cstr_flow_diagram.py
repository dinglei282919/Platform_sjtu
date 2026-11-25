"""Check rendered pipe motion against the applied flow and the session clock."""
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from desktop.risk_control.process_diagram import ProcessDiagram


class PipeDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.diagram = ProcessDiagram()
        self.diagram.setAttribute(Qt.WA_DontShowOnScreen)
        self.diagram.resize(760, 250)
        self.diagram.show()
        self.app.processEvents()
        self.addCleanup(self.diagram.close)
        timer = patch("desktop.risk_control.process_diagram.monotonic", return_value=10.)
        self.clock = timer.start()
        self.addCleanup(timer.stop)
        self.sample = dict(seq=1, T=432., Tc=418., C=.096, Qc=200., Qc_cmd=30.,
                           T_ref=433., Ci=1., Ti=350., Tci=350.)
        self.diagram.set_sample(self.sample, "first", True)

    def frame(self):
        return self.diagram.grab().toImage()

    @staticmethod
    def band(image, start, y, length=96):
        scale = image.width()/760
        return [image.pixelColor(round(x*scale), round(y*scale)).green() > 170
                for x in range(start, start+length)]

    def assert_shift(self, before, after, x, y, distance):
        a, b = self.band(before, x, y), self.band(after, x, y)
        self.assertGreater(sum(left != right for left, right in zip(a, b)), 12)
        errors = sum(left != right for left, right in zip(a[:-distance], b[distance:]))
        self.assertLessEqual(errors, 5, f"stripes at {(x, y)} should move right by {distance}")

    def test_material_and_both_coolant_legs_move_at_their_applied_rates(self):
        before = self.frame()
        self.clock.return_value += .25
        self.diagram._tick()
        after = self.frame()
        for x, y in ((52, 45), (520, 177)):
            self.assert_shift(before, after, x, y, 6)
        for x, y in ((52, 176), (520, 106)):
            self.assert_shift(before, after, x, y, 12)
        self.assertEqual(self.band(before, 228, 115, 83), self.band(after, 228, 115, 83))

        self.diagram.set_sample({**self.sample, "seq": 2, "Qc": 100.}, "first", True)
        current = self.frame()
        self.assertEqual(self.band(after, 52, 176), self.band(current, 52, 176))
        self.clock.return_value += .25
        self.diagram._tick()
        self.assert_shift(current, self.frame(), 52, 176, 6)

    def test_both_pipe_walls_remain_visible_around_the_moving_stripes(self):
        image = self.frame()
        scale = image.width()/760

        def green(x, y):
            return image.pixelColor(round(x*scale), round(y*scale)).green()

        for y in (45, 176):
            self.assertGreater(green(144, y-7), 235)
            wall = green(144, y-4)
            bore = green(144, y)
            self.assertLess(wall, 155)
            self.assertGreater(bore, wall+10)
            self.assertLess(bore, 170)

    def test_fdi_preview_effective_and_cancelled_lines_on_a_paused_sample(self):
        def marked_pixels(side, active, visible=True):
            marker = dict(link=side, active=active, label="FDI") if visible else {}
            sample = dict(self.sample, loop_side=side, signal_marker=marker, flow_read_previous=148.)
            self.diagram.set_sample(sample, "trial", False)
            image = self.frame()
            def marked(x, y):
                color = image.pixelColor(x,y)
                return color.red() > 1.5*color.green() and color.red()>120
            return (sum(marked(x,115) for x in range(226,308)),
                    sum(marked(207,y) for y in range(132,160)))
        for side, selected in (("measurement",0), ("execution",1)):
            preview = marked_pixels(side,False)
            active = marked_pixels(side,True)
            self.assertGreater(active[selected],preview[selected])
            self.assertGreater(preview[selected],0)
            self.assertEqual(active[1-selected],0)
            self.assertEqual(marked_pixels(side,False,visible=False),(0,0))

    def test_zero_coolant_pause_hide_resume_and_new_run(self):
        self.diagram.set_sample({**self.sample, "seq": 2, "Qc": 0.}, "first", True)
        before = self.frame()
        self.clock.return_value += .25
        self.diagram._tick()
        after = self.frame()
        self.assertEqual(self.band(before, 52, 176), self.band(after, 52, 176))
        self.assert_shift(before, after, 52, 45, 6)

        self.diagram.set_running(False)
        self.assertFalse(self.diagram._timer.isActive())
        self.clock.return_value += 5
        self.assertEqual(self.band(after, 52, 45), self.band(self.frame(), 52, 45))
        self.diagram.set_running(True)
        self.diagram.hide()
        self.assertFalse(self.diagram._timer.isActive())
        self.clock.return_value += 5
        self.diagram.show()
        self.assertTrue(self.diagram._timer.isActive())
        self.diagram._tick()
        self.assertEqual(self.band(after, 52, 45), self.band(self.frame(), 52, 45))
        self.diagram.set_sample(self.sample, "second", False)
        self.assertEqual(self.band(before, 52, 45), self.band(self.frame(), 52, 45))
        self.assertFalse(self.diagram._timer.isActive())

