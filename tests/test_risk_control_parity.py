"""Check actual desktop defaults against the same Web API contracts."""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from fastapi.testclient import TestClient
from web_backend.task_manager import TaskManager

# Collection must not open or repair the user's live task repository.
_runtime = TemporaryDirectory(prefix='control-parity-api-')
_init = TaskManager.__init__
with patch.object(TaskManager, '__init__', lambda self, runtime_root=None:
                  _init(self, runtime_root or Path(_runtime.name))):
    from web_backend.app import app

from platform_core.risk_control.dnn_mpc import config as dnn_config
from desktop.risk_control.process_control_dnn_mpc import ProcessControlDnnMpcWidget
from desktop.risk_control.fdi_page import FdiWidget
from desktop.risk_control.replay_page import ReplayWidget
from desktop.risk_control.self_healing_page import SelfHealingWidget
from desktop.risk_control.reactor_page import ReactorWidget


class RiskControlParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.client = TestClient(app)

    def test_training_and_mpc_default_payloads(self):
        with TemporaryDirectory() as temp, patch.object(dnn_config, 'LATEST_TRAINING_STATE_PATH', Path(temp) / 'absent.json'):
            for mode in ('training', 'mpc'):
                widget = ProcessControlDnnMpcWidget(mode)
                try:
                    defaults = self.client.get(f'/api/{mode}/defaults').json()
                    self.assertEqual(widget._config(), dnn_config.validate_config(mode, defaults))
                finally:
                    widget.close()

    def test_fdi_and_replay_duration_contracts(self):
        for mode, cls in [('fdi', FdiWidget), ('replay', ReplayWidget)]:
            widget = cls()
            try:
                defaults = self.client.get(f'/api/{mode}/defaults').json()
                self.assertEqual(widget.duration.value(), defaults['sim_time'])
                self.assertEqual(self.client.post(f'/api/{mode}/tasks', json={'sim_time': widget.duration.minimum() - .01}).status_code, 422)
                self.assertEqual(self.client.post(f'/api/{mode}/tasks', json={'sim_time': widget.duration.maximum() + .01}).status_code, 422)
            finally:
                widget.close()

    def test_all_self_healing_scenarios_and_parameters(self):
        catalog = self.client.get('/api/self-healing/defaults').json()
        widget = SelfHealingWidget()
        try:
            self.assertEqual(set(widget.fields), {p['key'] for p in catalog['parameters']})
            for index, (key, defaults) in enumerate(catalog['defaults'].items()):
                widget.scenario_tabs.setCurrentIndex(index)
                self.assertEqual(widget.scenario_tabs.tabData(index), key)
                self.assertEqual(widget.config(), defaults)
        finally:
            widget.close()

    def test_reactor_inputs_defaults_and_steps(self):
        defaults = self.client.get('/api/reactor/defaults').json()
        widget = ReactorWidget()
        try:
            self.assertEqual({key: field.value() for key, field in widget.inputs.items()}, defaults)
            steps = dict(duration=.1, sample_dt=.001, prescribed_time=.1, ell=.01,
                         x1=.05, x2=.05, observer1=.05, observer2=.05)
            self.assertEqual({key: field.singleStep() for key, field in widget.inputs.items()}, steps)
        finally:
            widget.close()
