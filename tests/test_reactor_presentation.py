"""Signed composition display and parameter provenance regressions."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from platform_core.risk_control.reactor.presentation import build_diagram_series, diagram_tokens, render_diagram
from platform_core.risk_control.reactor.coordinates import composition_series, scales
from platform_core.risk_control.reactor.core import Config, initial_state, derivatives
from platform_core.risk_control.reactor.kernel import derivatives_compiled

class ReactorPresentationTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(t=[0., .2], x1=[-.2,.5], x2=[.1,-.5], observer1=[0.,.2], observer2=[.2,.4], u=[-10.,20.],mode=[2,1])

    def test_inverse_transform_sign_and_estimation_errors(self):
        d=composition_series(self.data)
        np.testing.assert_allclose(d['xs'],[-.4,1.])
        np.testing.assert_allclose(d['xd'],[.1,-.5])
        np.testing.assert_allclose(d['error_s'],[-.4,.6])
        np.testing.assert_allclose(d['received_d'],d['k']*d['observer_d'])
        self.assertEqual(diagram_tokens(self.data)['XS'],'-0.4')

    def test_flows_belong_to_each_run_and_no_fictitious_levels_or_valve(self):
        d={**self.data,'r1':[.6,.6],'r2':[.4,.4]}
        series=build_diagram_series(d)
        self.assertEqual(series['r1'],[.6,.6])
        np.testing.assert_allclose(series['xs'],[-.3,.75])
        self.assertEqual(series['level1'],[.5,.5])
        svg=render_diagram(d)
        self.assertIn('0.600',svg)
        self.assertNotIn('开度',svg)
        self.assertNotIn('浓度100%',svg)
        self.assertNotIn('__',svg)
        ET.fromstring(svg)

    def test_parameterized_rhs_matches_original_composition_equations(self):
        for r1,r2 in ((.3,.2),(.6,.4),(.1,.7)):
            c=Config(r1=r1,r2=r2);y=initial_state(c); ss,sd=scales(r2)
            for mode in (1,2):
                dy=derivatives(.05,y,c,mode)
                np.testing.assert_allclose(dy,derivatives_compiled(.05,y,c.prescribed_time,c.ell,mode,r1,r2),rtol=1e-13,atol=1e-13)
                a=1/(.75 if mode==1 else 1)+(.2 if mode==1 else .4)
                b=1/(2 if mode==1 else 1.8)+.3-r2/.5
                from platform_core.risk_control.reactor.core import signals
                u=signals(.05,y,c,mode)['u']
                self.assertAlmostEqual(ss*dy[0],(1-r2)/.4*c.xd-a*c.xs)
                self.assertAlmostEqual(sd*dy[1],.5/.5*u+r1/.5*c.xs-b*c.xd)

    def test_transform_validation_and_fixed_physical_initial_conditions(self):
        for r2 in (-.1,1.,float('nan'),True):
            with self.assertRaises(ValueError):Config(r2=r2).validate()
        for r2 in (.2,.6):
            c=Config(r2=r2);y=initial_state(c);ss,sd=scales(r2)
            self.assertAlmostEqual(ss*y[0],1.)
            self.assertAlmostEqual(sd*y[1],.5)

    def test_animation_and_seeking_preserve_numeric_readouts(self):
        first=diagram_tokens(self.data,1)
        next_frame=diagram_tokens(self.data,1,phase=.3)
        for key in ('XS','XD','U','KAPPA','FLOW1','FLOW2'):
            self.assertEqual(first[key],next_frame[key])
        self.assertEqual(diagram_tokens()['XS'],'—')
        ET.fromstring(render_diagram())

    def test_saved_run_and_legacy_import_use_own_coordinates(self):
        import tempfile,json
        from dataclasses import asdict
        from pathlib import Path
        from platform_core.risk_control.reactor.core import simulate
        from platform_core.risk_control.reactor.reporting import export_result,load_result
        with tempfile.TemporaryDirectory() as temp:
            result=simulate(Config(duration=.1,prescribed_time=.1,r2=.4))
            export_result(result,temp);loaded=load_result(temp)
            np.testing.assert_allclose(loaded['data']['xs'],result['data']['x1']*1.5)
            self.assertEqual(loaded['config']['r2'],.4)
            self.assertIn('xs',Path(temp,'trajectory.csv').read_text().splitlines()[0].split(','))
            # An old export had transformed initial conditions and no flow fields.
            summary=Path(temp,'summary.json');old=json.loads(summary.read_text())
            old['config']={k:v for k,v in asdict(Config()).items() if k not in {'xs','xd','observer_s','observer_d','r1','r2'}}
            old['config'].update(duration=.1,prescribed_time=.1,x1=.5,x2=.5,observer1=0.,observer2=.5)
            summary.write_text(json.dumps(old))
            legacy=load_result(temp)
            np.testing.assert_allclose(legacy['data']['xs'],result['data']['x1']*2.)
            self.assertEqual(legacy['config']['xs'],1.)
            self.assertIsNone(legacy['config']['x1'])
