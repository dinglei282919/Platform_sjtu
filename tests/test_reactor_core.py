import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from platform_core.risk_control.reactor import Config, SimulationCancelled, simulate
from platform_core.risk_control.reactor.core import (CENTERS2, basis2, derivatives, gamma, initial_state,
                          jacobian, nussbaum, signals, switching_mode)
from platform_core.risk_control.reactor.reporting import export_result, load_result


class CoreTests(unittest.TestCase):
    def test_switch_edges_match_matlab(self):
        for edge, before, at, after in [(0.2,2,1,1), (0.7,1,2,2), (1.2,2,1,1),
                                       (1.6,1,2,2), (1.9,2,1,1), (2.3,1,1,2)]:
            self.assertEqual([switching_mode(t) for t in (edge-1e-8, edge, edge+1e-8)], [before,at,after])

    def test_initial_conditions(self):
        y = initial_state(Config())
        np.testing.assert_array_equal(y[:4], [.5,.5,0,.5])
        self.assertEqual(len(y), 37)
        np.testing.assert_array_equal(y[4:18], np.full(14,.5))
        np.testing.assert_array_equal(y[19:33], np.ones(14))
        np.testing.assert_array_equal(y[34:], [0,0,.75])

    def test_matlab_column_major_indexing(self):
        source_matrix = np.tile(np.arange(-3,4)*5, (2,1))
        np.testing.assert_array_equal(CENTERS2, source_matrix.flatten(order="F")[:7])
        manual = [np.exp(-np.linalg.norm(np.array([1.,2.])-d)**2/50) for d in [-15,-15,-10,-10,-5,-5,0]]
        np.testing.assert_allclose(basis2(1,2), manual)

    def test_gamma_continuity_and_cap(self):
        c = Config()
        self.assertEqual(gamma(0,c),1)
        self.assertAlmostEqual(gamma(1.5,c),np.exp(10))
        self.assertEqual(gamma(1.500001,c),gamma(3,c))

    def test_plant_and_observer_initial_derivatives(self):
        c = Config()
        y = initial_state(c)
        dy = derivatives(0,y,c,2)
        self.assertAlmostEqual(dy[0],-.2)
        self.assertAlmostEqual(dy[1],-.5*(1/1.8+.3-.4)+.6)
        self.assertAlmostEqual(dy[2],.5+np.exp(-((.5-np.arange(-3,4)*5)**2)/50).sum()*.5+2.5)
        self.assertEqual(dy[36],0)

    def test_source_wiring_scaled_signals(self):
        c = Config()
        y = initial_state(c)
        y[34] = .25
        s = signals(0,y,c,2)
        self.assertEqual(s["kx1"],1)
        self.assertEqual(s["kx2"],1)
        self.assertEqual(s["v1"],1)
        self.assertEqual(s["v2"],.75)

    def test_inactive_weight_decay(self):
        c = Config()
        dy = derivatives(0,initial_state(c),c,2)
        np.testing.assert_array_equal(dy[4:11],np.full(7,-1.5))
        np.testing.assert_array_equal(dy[19:26],np.full(7,-3))

    def test_reject_invalid_parameters(self):
        for kwargs in ({"duration":4}, {"sample_dt":0}, {"x1":float("nan")},
                       {"ell":.001}, {"method":"RK45"}, {"prescribed_time":0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs).validate()

    def test_no_silent_nussbaum_clipping(self):
        with self.assertRaises(FloatingPointError):
            nussbaum(27)

    def test_complex_jacobian_against_directional_difference(self):
        c = Config()
        y = initial_state(c)
        direction = np.random.default_rng(123).normal(size=37)
        step = 1e-7
        central = (derivatives(.05,y+step*direction,c,2)-derivatives(.05,y-step*direction,c,2))/(2*step)
        np.testing.assert_allclose(jacobian(.05,y,c,2)@direction,central,rtol=1e-6,atol=2e-5)

    def test_batched_jacobian_matches_column_reference(self):
        from unittest.mock import patch
        from platform_core.risk_control.reactor import core
        c = Config()
        with np.load(Path(__file__).resolve().parent/"fixtures/reactor_jacobian.npz") as saved:
            for t in (0., .2, 1.5, 2.3):
                i = np.argmin(abs(saved["t"] - t))
                y = saved["states"][:,i]
                mode = switching_mode(t)
                h = 1e-25
                reference = np.column_stack([
                    derivatives(t,y.astype(complex)+1j*h*np.eye(37)[j],c,mode).imag/h
                    for j in range(37)])
                with patch.object(core,"derivatives",wraps=core.derivatives) as counted:
                    actual = jacobian(t,y,c,mode)
                    self.assertEqual(counted.call_count,1)
                np.testing.assert_allclose(actual,reference,rtol=2e-12,atol=1e-8)

    def test_cancel(self):
        with self.assertRaises(SimulationCancelled):
            simulate(cancelled=lambda: True)

    def test_compiled_rhs_matches_reference_across_modes_and_gain(self):
        from platform_core.risk_control.reactor.kernel import derivatives_compiled
        c = Config()
        with np.load(Path(__file__).resolve().parent/"fixtures/reactor_jacobian.npz") as saved:
            for t in (0., .2, 1.5, 2.3):
                y = saved["states"][:, np.argmin(abs(saved["t"] - t))]
                for mode in (1, 2):
                    for ell in (.15, .3):
                        config = Config(ell=ell)
                        np.testing.assert_allclose(
                            derivatives_compiled(t, y, config.prescribed_time, ell, mode),
                            derivatives(t, y, config, mode), rtol=3e-13, atol=1e-10)
        y = initial_state(c)
        y[36] = 27
        with self.assertRaises(FloatingPointError):
            derivatives_compiled(0., y, c.prescribed_time, c.ell, 2)

    def test_compiled_trajectory_matches_reference(self):
        for kwargs in (dict(duration=.3, prescribed_time=.2),
                       dict(duration=.4, prescribed_time=.3, x1=-.3, observer2=.2)):
            config = Config(**kwargs, rtol=1e-7, atol=1e-9)
            compiled = simulate(config)
            reference = simulate(config, compiled=False)
            np.testing.assert_allclose(compiled["states"], reference["states"], rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(compiled["data"]["u"], reference["data"]["u"], rtol=2e-4, atol=2e-6)

    def test_short_solver_agreement_and_export(self):
        kw = dict(duration=.3,prescribed_time=.2,ell=.15,rtol=1e-6,atol=1e-8)
        a, b = simulate(Config(**kw,method="Radau")), simulate(Config(**kw,method="BDF"))
        np.testing.assert_allclose(a["states"],b["states"],rtol=3e-4,atol=1e-5)
        self.assertTrue(np.all(np.diff(a["data"]["t"])>0))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            export_result(a,folder)
            summary = json.loads((Path(folder)/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["config"]["duration"],.3)
            self.assertEqual(set(np.load(Path(folder)/"trajectory.npz").files),set(a["data"])|{"states"})
            self.assertTrue((Path(folder)/"results.png").stat().st_size>1000)
            restored = load_result(folder)
            np.testing.assert_array_equal(restored["states"],a["states"])
            archive = Path(folder)/"trajectory.npz"
            np.savez(archive,**{**a["data"],"t":a["data"]["t"]*0,"states":a["states"]})
            with self.assertRaises(ValueError):
                load_result(folder)


if __name__ == "__main__":
    unittest.main()
