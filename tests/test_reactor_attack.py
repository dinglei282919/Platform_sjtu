import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from platform_core.risk_control.reactor.attack import multiplier,parse_attack
from platform_core.risk_control.reactor.core import Config,initial_state,signals,derivatives,simulate
from platform_core.risk_control.reactor.kernel import derivatives_compiled
from platform_core.risk_control.reactor.live import ReactorLiveSession
from platform_core.risk_control.reactor.presentation import diagram_tokens

class ReactorAttackTests(unittest.TestCase):
    def test_paper_defaults_and_positive_gain_bound(self):
        for t in np.linspace(0,100,101):
            self.assertAlmostEqual(multiplier(t),2+1.5*np.sin(.25*t))
            self.assertGreaterEqual(multiplier(t,1.8,.9,.7),.1-1e-14)
            self.assertEqual(multiplier(t,0,.9,.7),1.)

    def test_numpy_and_compiled_attack_variants_match(self):
        for scale,freq,phase in ((0,.25,0),(1,.25,0),(1.8,.8,-.7)):
            c=Config(attack_scale=scale,attack_frequency=freq,attack_phase=phase)
            c.validate();y=initial_state(c)
            for mode in (1,2):
                np.testing.assert_allclose(derivatives(.13,y,c,mode),derivatives_compiled(.13,y,c.prescribed_time,c.ell,mode,c.r1,c.r2,scale,freq,phase),rtol=1e-12,atol=1e-10)
                self.assertAlmostEqual(signals(.13,y,c,mode)['kx2'],multiplier(.13,scale,freq,phase)*y[3])

    def test_online_attack_preserves_states_clock_and_prior_history(self):
        s=ReactorLiveSession();s.advance(.1);s.advance(.1)
        old=s.state.copy();before=s.snapshot();t=s.t
        s.apply_event('attack',dict(scale=1.7,frequency=.6,phase=.4))
        np.testing.assert_array_equal(s.state,old);self.assertEqual(s.t,t)
        after=s.snapshot();self.assertEqual(after['data']['k'][:-1],before['data']['k'][:-1])
        self.assertAlmostEqual(after['data']['k'][-1],multiplier(t,1.7,.6,.4))
        after['events'][-1]['value']['scale']=999
        self.assertEqual(s.snapshot()['events'][-1]['value']['scale'],1.7)
        s.apply_event('attack_enabled',False)
        self.assertEqual(s.snapshot()['data']['k'][-1],1.)
        np.testing.assert_array_equal(s.state,old)
        s.advance(.1);s.apply_event('attack_enabled',True)
        self.assertAlmostEqual(s.snapshot()['data']['k'][-1],multiplier(s.t,1.7,.6,.4))

    def test_attack_changes_real_closed_loop_response(self):
        c=Config(duration=.2,prescribed_time=.1,ell=.3)
        off=simulate(replace(c,attack_scale=0.));on=simulate(replace(c,attack_scale=1.6,attack_frequency=.6,attack_phase=.3))
        self.assertGreater(np.linalg.norm(on['data']['u']-off['data']['u']),1.)
        self.assertGreater(np.linalg.norm(on['data']['xs']-off['data']['xs']),.001)
        np.testing.assert_allclose(on['data']['received_s'],on['data']['k']*on['data']['xs'])

    def test_zero_feedback_is_not_an_additive_disturbance(self):
        y=np.zeros(37);y[36]=.75
        c=Config(attack_scale=1.8)
        self.assertEqual(signals(1.,y,c)['u'],0.)
        np.testing.assert_array_equal(derivatives(1.,y,c,2),np.zeros(37))

    def test_invalid_and_failed_events_are_transactional(self):
        s=ReactorLiveSession();before=s.snapshot()
        for value in (dict(scale=1.81,frequency=.2,phase=0.),dict(scale=1.,frequency=.2,phase=4.),dict(scale='1',frequency=.2,phase=0.),dict(scale=1.)):
            with self.assertRaises(ValueError):s.apply_event('attack',value)
        with patch.object(s,'_record',side_effect=FloatingPointError('rejected')):
            with self.assertRaises(FloatingPointError):s.apply_event('attack',dict(scale=.4,frequency=.5,phase=.2))
        self.assertEqual(s.snapshot(),before)
        self.assertEqual(s.config.attack_scale,1.)

    def test_u_bar_is_signed_display_only_and_keeps_actual_number(self):
        def tokens(u):return diagram_tokens(dict(t=[0.],x1=[.5],x2=[.5],u=[u],mode=[2]))
        zero,neg,pos,large=tokens(0),tokens(-10),tokens(10),tokens(100)
        self.assertEqual(float(zero['U_BAR_WIDTH']),0.)
        self.assertLess(float(neg['U_BAR_X']),float(pos['U_BAR_X']))
        self.assertEqual(float(neg['U_BAR_WIDTH']),float(pos['U_BAR_WIDTH']))
        self.assertEqual(float(large['U_BAR_WIDTH']),180.)
        self.assertEqual(float(large['U']),100.)
