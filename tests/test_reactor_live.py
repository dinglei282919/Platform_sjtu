"""Incremental FDI regulator: real dynamics, parameter restarts, transactional failure."""
import unittest
from unittest.mock import patch
import numpy as np
from platform_core.risk_control.reactor.live import ReactorLiveSession
from platform_core.risk_control.reactor.core import SimulationCancelled

class ReactorLiveTests(unittest.TestCase):
    def test_initial_conditions_real_integration_and_fdi(self):
        s=ReactorLiveSession(); initial=s.snapshot()
        self.assertEqual(initial['data']['xs'],[1.])
        self.assertEqual(initial['data']['xd'],[.5])
        self.assertEqual(initial['data']['k'],[2.])
        self.assertEqual(s.snapshot(),initial)
        for _ in range(20):s.advance(.1)
        d=s.snapshot()['data']
        self.assertLess(abs(d['xs'][-1]),.05)
        self.assertTrue(np.isfinite(np.array(list(d.values()))).all())
        np.testing.assert_allclose(d['received_s'],np.array(d['k'])*d['xs'])
        np.testing.assert_allclose(d['received_d'],np.array(d['k'])*d['observer_d'])
        self.assertNotIn('disturbance',d)

    def test_parameter_restart_retains_physical_initial_values(self):
        s=ReactorLiveSession();s.advance(.1)
        s.apply_event('r2',.6)
        d=s.snapshot()['data']
        self.assertEqual(s.t,0.)
        self.assertEqual(d['r2'],[.6])
        self.assertEqual(d['xs'],[1.])
        self.assertAlmostEqual(s.state[0],1.)
        s.apply_event('r1',.5);s.advance(.1)
        self.assertEqual(s.config.r1,.5)
        self.assertTrue(all(v==.5 for v in s.snapshot()['data']['r1']))
        s.apply_event('reset');self.assertEqual(s.config.r2,.6)

    def test_control_off_holds_controller_states_and_actual_u_zero(self):
        s=ReactorLiveSession();s.advance(.1);s.apply_event('enabled',False)
        held=s.state[2:].copy();s.advance(.1)
        np.testing.assert_array_equal(s.state[2:],held)
        self.assertEqual(s.snapshot()['data']['u'][-1],0.)
        s.apply_event('enabled',True);s.advance(.1)

    def test_reject_old_disturbance_controls_and_invalid_parameters(self):
        s=ReactorLiveSession();before=s.snapshot()
        for kind,value in [('setpoint',.1),('disturbance',.1),('impulse',.1),('r2',1),('r1',-1),('r2',True),('mode',3),('enabled',1)]:
            with self.subTest(kind=kind,value=value),self.assertRaises(ValueError):s.apply_event(kind,value)
        self.assertEqual(s.snapshot(),before)

    def test_cancellation_timeout_and_failed_event_leave_state_unchanged(self):
        s=ReactorLiveSession();s.advance(.1);before=s.snapshot();state=s.state.copy()
        with self.assertRaises(SimulationCancelled):s.advance(.1,cancelled=lambda:True)
        with patch('platform_core.risk_control.reactor.live.time.perf_counter',side_effect=[0.,2.]):
            with self.assertRaises(RuntimeError):s.advance(.1)
        with patch.object(s,'_record',side_effect=FloatingPointError('failed record')):
            with self.assertRaises(FloatingPointError):s.apply_event('r2',.6)
        self.assertEqual(s.snapshot(),before);np.testing.assert_array_equal(s.state,state)

    def test_manual_mode_and_bounded_history(self):
        s=ReactorLiveSession();s.apply_event('mode',1)
        for _ in range(605):s.advance(.001)
        d=s.snapshot()['data'];self.assertEqual(len(d['t']),600)
        self.assertTrue(all(len(a)==600 for a in d.values()))
        self.assertEqual(d['mode'][-1],1)
