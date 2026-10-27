import copy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest

from desktop.risk_control.reactor_page import ReactorWidget, create_application
from desktop.risk_control.reactor_live_page import ReactorLiveWidget
from platform_core.risk_control.reactor.presentation import build_diagram_series


class FakeRuntime:
    """A cached worker boundary: the widget may command or snapshot, never solve."""

    def __init__(self):
        self.started = 0
        self.closed = False
        self.commands = []
        self.state = {
            "status": "running", "t": 1., "message": "",
            "controls": {"r1": .3, "r2": .2, "mode": 1, "enabled": True, "attack_enabled":True, "attack_scale":1., "attack_frequency":.25, "attack_phase":0.},
            "events": [],
            "data": {"t": [0., 1.], "x1": [0., .1], "x2": [0., .2], "u": [0., -.3],
                     "mode": [1, 1], "observer1": [0., .05], "observer2": [0., .1]},
        }
        self.state["data"].update(build_diagram_series(self.state["data"]))

    def start(self):
        self.started += 1

    def snapshot(self):
        return copy.deepcopy(self.state)

    def command(self, kind, value=None):
        self.commands.append((kind, value))
        if kind in self.state["controls"]:
            self.state["controls"][kind] = value
        elif kind in ("pause", "resume"):
            self.state["status"] = "paused" if kind == "pause" else "running"
        self.state["events"].append({"t": self.state["t"], "kind": kind, "value": value})

    def close(self, timeout=2):
        self.closed = True
        self.state["status"] = "closed"
        return True


class ReactorLiveDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_application()
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.runtime = FakeRuntime()
        self.widget = ReactorLiveWidget(runtime_factory=lambda: self.runtime)
        self.widget.resize(1300, 980)
        self.widget.show()
        self.app.processEvents()

    def tearDown(self):
        self.widget.shutdown()
        self.widget.close()
        self.app.processEvents()

    def test_open_starts_cached_worker_and_applies_commands_explicitly(self):
        w = self.widget
        self.assertEqual(self.runtime.started, 1)
        self.assertTrue(w.timer.isActive())
        self.assertEqual(w.diagram.sample_index, 1)
        w.inputs['r1'].setValue(.6)
        self.assertEqual(self.runtime.state['controls']['r1'],.3)
        w.apply_buttons['r1'].click()
        w.inputs['r2'].setValue(.4);w.apply_buttons['r2'].click()
        w.mode.setCurrentIndex(2);w.mode_button.click();w.enabled.click();w.refresh()
        self.assertIn(('r1',.6),self.runtime.commands)
        self.assertIn(('r2',.4),self.runtime.commands)
        self.assertIn(('mode',2),self.runtime.commands)
        self.assertIn(('enabled',False),self.runtime.commands)
        self.assertIn('0.600',w.applied.text())
        self.assertIn('R1',w.events.toPlainText())

    def test_snapshot_updates_plot_and_diagram_without_restarting_animation(self):
        w = self.widget
        w.diagram._advance_frame()
        phase = w.diagram.frame_index
        self.runtime.state["t"] = 2.
        data = self.runtime.state["data"]
        data["t"][-1], data["x1"][-1], data["u"][-1] = 2., .05, -.7
        data.update(build_diagram_series(data))
        w.refresh()
        self.assertEqual(w.diagram.frame_index, phase)
        self.assertEqual(w.diagram.data["u"][-1], -.7)
        lines=dict(w.plot_lines)
        self.assertEqual(list(lines['xs'].get_ydata()),[0.,.1])
        self.assertEqual(list(lines['u'].get_ydata()),[0.,-.7])
        self.assertIn("2.00 s", w.status.text())

    def test_manual_pause_survives_hide_show_and_close_ends_worker(self):
        w = self.widget
        w.toggle_pause()
        w.hide()
        self.app.processEvents()
        resumes = sum(kind == "resume" for kind, _ in self.runtime.commands)
        w.show()
        self.app.processEvents()
        self.assertEqual(sum(kind == "resume" for kind, _ in self.runtime.commands), resumes)
        self.assertEqual(w.snapshot_status(), "paused")
        w.toggle_pause()
        self.assertEqual(w.snapshot_status(), "running")
        w.hide()
        self.assertEqual(w.snapshot_status(), "paused")
        w.show()
        self.app.processEvents()
        self.assertEqual(w.snapshot_status(), "running")
        self.assertTrue(w.shutdown())
        self.assertTrue(self.runtime.closed)
        self.assertFalse(w.timer.isActive())

    def test_paused_command_updates_curves_at_the_same_simulation_time(self):
        w=self.widget;w.toggle_pause()
        data=self.runtime.state['data'];original_time=data['t'][-1]
        data['x1'][-1]=.3;data.update(build_diagram_series(data));w.refresh()
        line=dict(w.plot_lines)['xs']
        self.assertEqual(line.get_xdata()[-1],original_time)
        self.assertAlmostEqual(line.get_ydata()[-1],.6)

    def test_parameter_arrows_step_and_preserve_the_supported_range(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox
        for name, box in self.widget.inputs.items():
            with self.subTest(parameter=name):
                option = QStyleOptionSpinBox()
                box.initStyleOption(option)
                up = box.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, box)
                down = box.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, box)
                box.setValue(0)
                QTest.mouseClick(box, Qt.LeftButton, pos=up.center())
                self.assertAlmostEqual(box.value(), .05)
                QTest.mouseClick(box, Qt.LeftButton, pos=down.center())
                self.assertAlmostEqual(box.value(), 0)
                box.setValue(box.maximum())
                QTest.mouseClick(box, Qt.LeftButton, pos=up.center())
                self.assertAlmostEqual(box.value(), box.maximum())
                QTest.mouseClick(box, Qt.LeftButton, pos=down.center())
                self.assertAlmostEqual(box.value(), box.maximum()-.05)
        self.assertIn("R1/R2", self.widget.range_hint.text())

    def test_attack_fields_send_atomic_parameters_and_toggle(self):
        w=self.widget
        w.attack_inputs['scale'].setValue(1.5)
        w.attack_inputs['frequency'].setValue(.7)
        w.attack_inputs['phase'].setValue(-.4)
        w.attack_apply.click()
        self.assertIn(('attack',dict(scale=1.5,frequency=.7,phase=-.4)),self.runtime.commands)
        w.attack_toggle.click()
        self.assertIn(('attack_enabled',False),self.runtime.commands)

    def test_wrapper_blocks_online_switch_while_offline_worker_is_busy(self):
        runtime = FakeRuntime()
        wrapper = ReactorWidget(runtime_factory=lambda: runtime)
        wrapper.show()
        self.app.processEvents()
        try:
            self.assertEqual(wrapper.tabs.currentIndex(), 0)
            wrapper.tabs.setCurrentIndex(1)
            self.app.processEvents()
            wrapper._sync_modes()
            self.assertEqual(runtime.state["status"], "paused")
            self.assertTrue(wrapper.experiment.isEnabled())
            self.assertTrue(wrapper.experiment.diagram.animated)
            wrapper.hide()
            self.app.processEvents()
            wrapper.show()
            self.app.processEvents()
            self.assertTrue(wrapper.experiment.diagram.animated)
            wrapper.experiment.worker = object()
            wrapper.experiment.set_busy(True)
            self.assertTrue(wrapper.has_running_worker())
            self.assertFalse(wrapper.tabs.isTabEnabled(0))
            wrapper.tabs.setCurrentIndex(0)
            self.assertEqual(wrapper.tabs.currentIndex(), 1)
            wrapper.experiment.worker = None
            wrapper.experiment.set_busy(False)
            wrapper.tabs.setCurrentIndex(0)
            self.app.processEvents()
            self.assertEqual(runtime.state["status"], "running")
            self.assertFalse(wrapper.has_running_worker())
        finally:
            wrapper.experiment.worker = None
            wrapper.shutdown()
            wrapper.close()
            self.app.processEvents()
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
