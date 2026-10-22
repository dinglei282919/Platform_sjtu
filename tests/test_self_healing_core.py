"""Core self-healing contracts: algorithms, physical models, batching and reports."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from platform_core.risk_control.self_healing.algorithms import ARReference, CCSC, SCSC, Standardizer
from platform_core.risk_control.self_healing.cstr import CSTRPlant
from platform_core.risk_control.self_healing.metrics import evaluate
from platform_core.risk_control.self_healing.reporting import save_result
from platform_core.risk_control.self_healing.simulation import ExperimentConfig, collect_training, defaults, run_experiment
from platform_core.risk_control.self_healing.water import WaterPlant


# Control algorithms

class StandardizerTests(unittest.TestCase):
    def test_sample_normalization_and_constant_column(self):
        x = np.array([[1.0, 7.0], [3.0, 7.0], [5.0, 7.0]])
        scaler = Standardizer().fit(x)
        z = scaler.transform(x)
        np.testing.assert_allclose(z[:, 0], [-1, 0, 1])
        np.testing.assert_array_equal(z[:, 1], 0)
        np.testing.assert_allclose(scaler.inverse_transform(z), x)
        np.testing.assert_array_equal(scaler.active, [True, False])


class SCSCTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 30], [2, 20], [3, 10]], dtype=float)
        self.y = np.array([[10, 3], [20, 2], [30, 1]], dtype=float)

    def test_eq17_uses_weights_current_input_mean_and_row_vectors(self):
        model = SCSC(weights=[1, 0.25]).fit(self.x, self.y)
        np.testing.assert_allclose(model.correlation, [[1, -1], [-1, 1]])
        batch = np.array([[22, 2.2], [26, 2.6]])
        # Deviation = [0.2, 0.2]; weighted = [0.2, 0.05].
        # @ P.T -> [0.15, -0.15]; current input mean = [4, 40].
        current_x = np.array([[3, 30], [5, 50]])
        np.testing.assert_allclose(model.correction(batch, current_x=current_x), [0.6, -6])
        np.testing.assert_allclose(model.correction_batch(batch, current_x=current_x), [[0.6, -6]] * 2)
        # Gain/sign belongs to the caller, not the algorithm.
        np.testing.assert_allclose(model.correction(batch), [0.3, -3])

    def test_nominal_output_means_produce_no_adjustment(self):
        np.testing.assert_allclose(SCSC().fit(self.x, self.y).correction(self.y), 0)

    def test_zero_reference_and_constant_training_are_finite(self):
        model = SCSC().fit(self.x, self.y)
        self.assertTrue(np.isfinite(model.correction(self.y, reference_y=[0, -1e-20])).all())
        self.assertEqual(model.diagnostics()["last_guarded_outputs"], [0, 1])
        constant_model = SCSC().fit(np.ones((5, 2)), np.ones((5, 3)))
        np.testing.assert_array_equal(constant_model.correction(np.ones((2, 3)) * 10), 0)


class CCSCTests(unittest.TestCase):
    def test_canonical_scores_and_noisy_regression_in_physical_units(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(300, 2)) * [0.02, 40] + [1, 350]
        y = x @ np.array([[8., -3., 2.], [.1, .4, -.2]])
        y += rng.normal(size=y.shape) * [0.2, 2, 4]
        model = CCSC(ridge=0).fit(x, y)
        xz = model.x_scaler.transform(x)
        yz = model.y_scaler.transform(y)
        scores_x, scores_y = xz @ model.j, yz @ model.l
        np.testing.assert_allclose(scores_x.T @ scores_x / 299, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(scores_y.T @ scores_y / 299, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(scores_x.T @ scores_y / 299, model.sigma, atol=1e-10)
        # Independent least-squares calculation uses physical, centered data;
        # no CCA/SVD matrices from the controller enter this reference.
        regression = np.linalg.lstsq(y - y.mean(0), x - x.mean(0), rcond=None)[0]
        deviation = np.array([.4, -1., 3.])
        np.testing.assert_allclose(model.correction([y.mean(0) + deviation]),
                                   deviation @ regression, atol=1e-9)

    def test_full_rank_mapping_matches_independent_covariance_identity(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=(200, 3)) * [2, 7, 0.4] + [20, 350, 5]
        transform = np.array([[2, -1], [0.3, 0.7], [1.5, 0.2]])
        x = (y - y.mean(axis=0)) @ transform + [10, 100]
        model = CCSC(ridge=0).fit(x, y)
        deviation = np.array([0.4, -2.0, 0.7])
        # Eq. 24 reduces to C_yx @ inv(C_yy) in column form.
        # For x = y @ transform, it must recover deviation @ transform.
        np.testing.assert_allclose(model.correction([model.y_mean + deviation]), deviation @ transform, atol=1e-10)
        self.assertTrue(np.all((model.canonical_correlations >= 0) & (model.canonical_correlations <= 1)))

    def test_literal_batch_centering_has_zero_mean_but_nonzero_waveform(self):
        x = np.arange(1.0, 11.0)[:, None]
        y = 3 * x + 7
        literal = CCSC(centering="batch", ridge=0).fit(x, y)
        batch = np.array([[100.0], [103.0], [106.0]])
        np.testing.assert_allclose(literal.correction_batch(batch), [[-1], [0], [1]], atol=1e-12)
        np.testing.assert_allclose(literal.correction(batch), [0], atol=1e-12)
        self.assertTrue(literal.diagnostics()["literal_equation_22"])
        engineering = CCSC(centering="reference", ridge=0).fit(x, y)
        self.assertGreater(abs(engineering.correction(batch)[0]), 1)

    def test_rank_deficient_and_constant_columns_stay_finite(self):
        t = np.arange(1.0, 21.0)
        x = np.column_stack((t, 2 * t, np.ones_like(t)))
        y = np.column_stack((3 * t, 6 * t, np.zeros_like(t), np.ones_like(t)))
        for ridge in (0, 1e-8):
            model = CCSC(ridge=ridge).fit(x, y)
            delta = model.correction(y + [3, 6, 100, 100])
            self.assertTrue(np.isfinite(delta).all())
            np.testing.assert_allclose(delta, [1, 2, 0], atol=1e-6)
            self.assertEqual(model.diagnostics()["x_rank"], 1)

    def test_unequal_input_output_dimensions(self):
        rng = np.random.default_rng(24)
        for m, p in ((4, 2), (2, 4)):
            x, y = rng.normal(size=(30, m)), rng.normal(size=(30, p))
            model = CCSC().fit(x, y)
            self.assertEqual(model.correction_batch(y[:7]).shape, (7, m))
            self.assertEqual(model.correction(y[:7]).shape, (m,))


class ARReferenceTests(unittest.TestCase):
    def test_recovers_known_lag_law_and_does_not_mutate_history(self):
        sequence = [1.0, 1.8]
        for _ in range(38):
            sequence.append(0.7 * sequence[-1] + 0.2 * sequence[-2])
        values = np.column_stack((sequence, np.asarray(sequence) * 3))
        model = ARReference(order=2, ridge=0).fit(values)
        np.testing.assert_allclose(model.coefficients, [[0.7, 0.7], [0.2, 0.2]], atol=1e-12)
        expected = 0.7 * values[-1] + 0.2 * values[-2]
        original_history = model.history.copy()
        np.testing.assert_allclose(model.predict(), expected)
        np.testing.assert_allclose(model.predict(), expected)
        np.testing.assert_allclose(model.predict(2), 0.7 * expected + 0.2 * values[-1])
        np.testing.assert_array_equal(model.history, original_history)

    def test_constant_and_zero_normal_reference(self):
        model = ARReference(order=3).fit(np.tile([0.0, 350.0], (20, 1)))
        np.testing.assert_allclose(model.predict(), [0, 350], atol=1e-8)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ARReference(order=0)
        with self.assertRaises(ValueError):
            ARReference(order=3).fit(np.ones((3, 2)))
        with self.assertRaises(RuntimeError):
            ARReference().predict()
        with self.assertRaises(ValueError):
            SCSC().fit(np.ones((3, 2)), np.ones((4, 2)))
        with self.assertRaises(ValueError):
            CCSC(centering="implicit")
        with self.assertRaises(ValueError):
            SCSC().fit([[1], [float("nan")]], [[1], [2]])


# Three-tank model

class TestWaterPlant(unittest.TestCase):
    def test_nominal_equilibrium_and_mass_balance(self):
        plant = WaterPlant()
        for i in range(20):
            output = plant.step(plant.nominal_input, i * plant.dt)
        np.testing.assert_allclose(plant.state, [50, 50, 50], atol=1e-11)
        np.testing.assert_allclose(output, plant.reference_output, atol=1e-11)
        self.assertAlmostEqual(output[0], sum(output[1:]))
        h = np.array([25.0, 40.0, 60.0])
        qin = np.array([15.0, 20.0, 10.0])
        np.testing.assert_allclose(plant.derivative(h, qin, 0) * plant.area,
                                   qin - plant.flow_rates(0, h))
        np.testing.assert_array_equal(plant.inlet_flow(np.zeros(3)), np.zeros(3))

    def test_blockage_affects_flow_and_mass_balance(self):
        plant = WaterPlant(fault="blockage", fault_start=2, fault_duration=10)
        healthy_flow = plant.flow_rates(2)
        blocked_flow = plant.flow_rates(12)
        np.testing.assert_array_equal(blocked_flow[:2], healthy_flow[:2])
        self.assertAlmostEqual(blocked_flow[2], 0.4 * healthy_flow[2])
        before = plant.derivative(plant.state, healthy_flow, 2)
        after = plant.derivative(plant.state, healthy_flow, 12)
        self.assertEqual(before[2], 0.0)
        self.assertGreater(after[2], 0.0)
        for i in range(300):
            y = plant.step(plant.nominal_input, i * plant.dt)
        np.testing.assert_allclose(plant.state, plant.nominal_input, atol=0.05)
        self.assertLess(y[3], 0.41 * plant.reference_output[3])
        self.assertLess(y[0], plant.reference_output[0])

    def test_step_tracking_and_step_size_convergence(self):
        coarse = WaterPlant(dt=0.1, max_internal_step=0.05)
        fine = WaterPlant(dt=0.05, max_internal_step=0.01)
        requested = np.array([60, 55, 65])
        for i in range(200):
            coarse.step(requested, i * coarse.dt)
            fine.step(requested, (2 * i) * fine.dt)
            fine.step(requested, (2 * i + 1) * fine.dt)
        np.testing.assert_allclose(coarse.state, fine.state, atol=1e-6)
        np.testing.assert_allclose(coarse.state, requested, atol=0.1)

    def test_bounds_reset_and_validation(self):
        plant = WaterPlant(dt=0.5, fault="blockage", fault_start=0, fault_duration=1)
        for i in range(30):
            y = plant.step(np.array([-100, 500, 500]), i * plant.dt)
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertTrue(np.all((plant.state >= 0) & (plant.state <= 100)))
            self.assertTrue(np.all((plant.valve_opening >= 0) & (plant.valve_opening <= 100)))
        np.testing.assert_array_equal(plant.last_setpoints, [5, 95, 95])
        np.testing.assert_allclose(plant.reset(), plant.reference_output)
        np.testing.assert_array_equal(plant.state, plant.initial_state)
        with self.assertRaises(ValueError):
            WaterPlant(dt=0)
        with self.assertRaises(ValueError):
            WaterPlant(fault="unknown")
        with self.assertRaises(ValueError):
            plant.step([50, float("nan"), 50], 0)
        with self.assertRaises(ValueError):
            plant.step([50, 50], 0)
        with self.assertRaises(ValueError):
            plant.reset([-1, 50, 50])


# CSTR model

class CSTRPlantTests(unittest.TestCase):
    def test_nominal_state_is_an_equilibrium(self):
        plant = CSTRPlant()
        for time in range(30):
            state = plant.step(plant.nominal_input, time)
        np.testing.assert_allclose(state, plant.reference_output, rtol=0, atol=1e-8)
        self.assertAlmostEqual(plant.cooling_flow, plant.nominal_cooling_flow, places=7)

    def test_source_fault_profiles_and_custom_onset(self):
        htc = CSTRPlant(fault="htc")
        catalyst = CSTRPlant(fault="catalyst", fault_start=160)
        self.assertEqual(htc.fault_factors(199), (1.0, 1.0))
        self.assertEqual(htc.fault_factors(200), (1.0, 1.0))
        self.assertAlmostEqual(htc.fault_factors(1200)[1], math.exp(-1))
        self.assertAlmostEqual(catalyst.fault_factors(1160)[0], math.exp(-0.5))
        self.assertEqual(catalyst.fault_factors(1160)[1], 1.0)

    def test_heat_transfer_damage_demands_more_cooling_flow(self):
        # Lower heat transfer initially raises T; the PI increases cooling flow.
        plant = CSTRPlant(fault="htc", fault_start=0)
        for time in range(60):
            plant.step(plant.nominal_input, time)
        self.assertGreater(plant.cooling_flow, plant.nominal_cooling_flow)
        self.assertTrue(np.isfinite(plant.state).all())

    def test_integration_converges_under_catalyst_fault(self):
        coarse = CSTRPlant(fault="catalyst", fault_start=0, catalyst_initial_factor=0.9)
        fine = CSTRPlant(fault="catalyst", fault_start=0, catalyst_initial_factor=0.9, integration_step=0.005)
        for time in range(10):
            a = coarse.step(coarse.nominal_input, time)
            b = fine.step(fine.nominal_input, time)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_reset_shape_timing_and_input_bounds(self):
        plant = CSTRPlant(input_mode="legacy")
        self.assertEqual(len(plant.input_names), 4)
        with self.assertRaises(ValueError):
            plant.step([1, 350, 350], 0)
        with self.assertRaises(ValueError):
            plant.step(plant.nominal_input, 1)
        plant.step(plant.nominal_input, 0)
        np.testing.assert_allclose(plant.reset(), plant.reference_output)
        self.assertEqual(plant.time, 0)
        # A small excursion past a physical command limit must be clipped.
        command = plant.nominal_input.copy()
        command[0] = 2.1
        plant.step(command, 0)
        self.assertEqual(plant.last_input[0], 2.0)
        self.assertGreaterEqual(plant.cooling_flow, 10)
        self.assertLessEqual(plant.cooling_flow, 200)

    def test_invalid_configuration(self):
        for kwargs in ({"dt": 0}, {"integration_step": 0.1}, {"fault": "unknown"}, {"fault_rate": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CSTRPlant(**kwargs)


# Batch simulation and reporting

class TinyAffinePlant:
    """Four outputs preserve water-controller priorities; no numerical solver."""

    nominal_input = np.array([10.0, 20.0])
    reference_output = np.array([10.0, 20.0, 30.0, 40.0])
    input_bounds = (np.array([1.0, 1.0]), np.array([50.0, 50.0]))
    input_names = ("u1", "u2")
    output_names = ("y1", "y2", "y3", "y4")
    mapping = np.array([[1.0, 0.3, 0.5, -0.2], [-0.2, 1.0, 0.4, 0.6]])

    def __init__(self, config, normal, zero_fault=False):
        self.normal = normal
        self.zero_fault = zero_fault
        self.fault_start = config.fault_start
        self.calls = []

    def step(self, command, time):
        command = np.asarray(command, dtype=float)
        self.calls.append((float(time), command.copy()))
        result = self.reference_output + (command - self.nominal_input) @ self.mapping
        if not self.normal and not self.zero_fault and time >= self.fault_start:
            result = result + [1.0, -2.0, 0.5, 1.0]
        return result


def tiny_config(**kwargs):
    config = ExperimentConfig(
        scenario="water", duration=10, dt=1, batch_duration=2, fault_start=2,
        training_batches=8, training_hold=4, excitation=0.05,
        detection_fraction=0.001, max_step_fraction=0.1,
        scsc_gain=-0.25, ccsc_gain=-0.25,
    )
    return replace(config, **kwargs)


class ConfigTests(unittest.TestCase):
    def test_fractional_sample_is_rejected_at_long_horizons(self):
        for changes in ({"duration": 1000.005, "dt": .1},
                        {"duration": 2000., "batch_duration": 1000.005, "dt": .1},
                        {"training_hold": 1000.005, "dt": .1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tiny_config(**changes).validate()
        # Decimal representation alone must not reject an integral sample grid.
        tiny_config(duration=.3, dt=.1, batch_duration=.1,
                    training_hold=.6, fault_start=.1).validate()

    def test_all_presets_are_valid(self):
        for scenario in ("water", "cstr_htc", "cstr_catalyst"):
            defaults(scenario).validate()

    def test_invalid_time_scales_and_domains_are_rejected(self):
        invalid = [
            {"duration": -1}, {"dt": 0}, {"duration": 10.5},
            {"batch_duration": 0.5}, {"training_hold": 1},
            {"fault_start": -1}, {"fault_start": 10},
            {"training_batches": 7}, {"training_batches": 8.5},
            {"excitation": 0}, {"excitation": 0.5},
            {"measurement_noise": -0.1}, {"max_step_fraction": 0},
            {"detection_fraction": 0}, {"scsc_gain": np.nan},
            {"scenario": "unknown"}, {"reference_mode": "adapt_fault"},
            {"ccsc_centering": "undocumented"}, {"cstr_input_mode": "unknown"},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tiny_config(**changes).validate()


class MetricTests(unittest.TestCase):
    def test_denominators_and_absolute_time_discrete_atae(self):
        actual = np.array([[2.0, 8.0], [8.0, 2.0]])
        reference = np.array([4.0, 4.0])
        metrics = evaluate(actual, reference, np.array([10.0, 20.0]))
        np.testing.assert_allclose(metrics["rmse"], [np.sqrt(10), np.sqrt(10)])
        np.testing.assert_allclose(metrics["mape_actual_percent"], [75, 75])
        np.testing.assert_allclose(metrics["mape_reference_percent"], [75, 75])
        # Absolute times, no subtraction of window start and no dt multiplier.
        np.testing.assert_allclose(metrics["atae"], [100, 80])
        asymmetric = evaluate([[2.0], [4.0]], [4.0], [10.0, 20.0])
        np.testing.assert_allclose(asymmetric["mape_actual_percent"], [50])
        np.testing.assert_allclose(asymmetric["mape_reference_percent"], [25])

    def test_zero_denominator_guard_and_invalid_metric_data(self):
        metrics = evaluate([[0.0, 2.0]], [1.0, 0.0], [1.0])
        self.assertTrue(all(np.isfinite(value).all() for value in metrics.values()))
        for actual, reference, times in [([], [1], []), ([[1]], [1], [1, 2]), ([[np.nan]], [1], [1])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                evaluate(actual, reference, times)


class SimulationTests(unittest.TestCase):
    def _factory(self, instances, zero_fault=False):
        def make(config, normal=False):
            process = TinyAffinePlant(config, normal, zero_fault=zero_fault)
            instances.append(process)
            return process
        return make

    def test_training_is_normal_only_and_uses_settled_half_of_each_hold(self):
        config = tiny_config(fault_start=0)
        instances = []
        with patch("platform_core.risk_control.self_healing.simulation.make_plant", side_effect=self._factory(instances)):
            x, y, batch_means = collect_training(config)
        self.assertEqual(len(instances), 1)
        self.assertTrue(instances[0].normal)
        self.assertEqual(x.shape, (config.training_batches * 2, 2))
        self.assertEqual(batch_means.shape, (config.training_batches, 4))
        np.testing.assert_allclose(y, TinyAffinePlant.reference_output + (x - TinyAffinePlant.nominal_input) @ TinyAffinePlant.mapping)
        for batch in range(config.training_batches):
            np.testing.assert_allclose(batch_means[batch], y[batch * 2:(batch + 1) * 2].mean(axis=0))
            np.testing.assert_array_equal(x[batch * 2], instances[0].calls[batch * 4 + 2][1])

    def test_batch_adjustment_is_applied_only_to_next_batch(self):
        config = tiny_config(fault_start=0, scsc_gain=-0.25, ccsc_gain=-0.5)
        instances, controller_instances = [], []

        class KnownAdjustment:
            def __init__(self, **kwargs):
                self.calls = []
                controller_instances.append(self)

            def fit(self, x, y):
                self.training_x, self.training_y = x.copy(), y.copy()
                return self

            def correction(self, y_batch, reference_y=None, current_x=None):
                self.calls.append((y_batch.copy(), reference_y.copy(), current_x.copy()))
                return np.array([1.0, 2.0])

            def diagnostics(self):
                return {"test": "known-adjustment"}

        with patch("platform_core.risk_control.self_healing.simulation.make_plant", side_effect=self._factory(instances)), \
             patch("platform_core.risk_control.self_healing.simulation.SCSC", KnownAdjustment), \
             patch("platform_core.risk_control.self_healing.simulation.CCSC", KnownAdjustment):
            result = run_experiment(config)
        for method, gain in (("scsc", -0.25), ("ccsc", -0.5)):
            run = result["runs"][method]
            np.testing.assert_allclose(run["inputs"][:2], [[10, 20]] * 2)
            np.testing.assert_allclose(run["inputs"][2:4], [np.array([10, 20]) + gain * np.array([1, 2])] * 2)
            self.assertEqual(run["events"][0]["time"], 2.0)
            self.assertEqual([event["time"] for event in run["events"]], [2, 4, 6, 8])
        self.assertEqual(controller_instances[0].calls[0][0].shape, (2, 4))
        np.testing.assert_array_equal(controller_instances[0].calls[0][2], [10, 20])
        # Training and runtime processes are separate objects and normal
        # training data never receive the synthetic post-fault offset.
        self.assertEqual([plant.normal for plant in instances], [True, True, True, False, False, False])
        np.testing.assert_allclose(controller_instances[0].training_y,
                                   TinyAffinePlant.reference_output + (controller_instances[0].training_x - TinyAffinePlant.nominal_input) @ TinyAffinePlant.mapping)

    def test_zero_fault_does_not_trigger_and_seed_controls_training_and_noise(self):
        config = tiny_config()
        with patch("platform_core.risk_control.self_healing.simulation.make_plant", side_effect=self._factory([], zero_fault=True)):
            normal = run_experiment(config)
        for run in normal["runs"].values():
            self.assertEqual(run["events"], [])
            np.testing.assert_allclose(run["outputs"], np.tile(TinyAffinePlant.reference_output, (10, 1)))

        noisy_config = replace(config, measurement_noise=0.002)
        with patch("platform_core.risk_control.self_healing.simulation.make_plant", side_effect=self._factory([])):
            first = run_experiment(noisy_config)
            second = run_experiment(noisy_config)
            other = run_experiment(replace(noisy_config, seed=noisy_config.seed + 10))
        np.testing.assert_array_equal(first["training"]["inputs"], second["training"]["inputs"])
        self.assertFalse(np.array_equal(first["training"]["inputs"], other["training"]["inputs"]))
        baseline_noise = first["runs"]["normal"]["measured"] - first["runs"]["normal"]["outputs"]
        for method in first["runs"]:
            np.testing.assert_array_equal(first["runs"][method]["measured"], second["runs"][method]["measured"])
            np.testing.assert_allclose(first["runs"][method]["measured"] - first["runs"][method]["outputs"], baseline_noise, atol=1e-14)

    def test_saved_artifacts_preserve_input_output_alignment_and_metrics(self):
        with patch("platform_core.risk_control.self_healing.simulation.make_plant", side_effect=self._factory([])):
            result = run_experiment(tiny_config())
        # Keep generated test artifacts inside the writable workspace. Some
        # Windows sandbox profiles expose a TEMP directory that denies writes.
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="test-report-", dir=test_root) as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(test_root))
            directory = save_result(result, temporary, plots=False)
            self.assertEqual(directory, Path(temporary) / "water")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"], result["metrics"])
            self.assertEqual(summary["events"]["scsc"], result["runs"]["scsc"]["events"])
            csv_data = np.loadtxt(directory / "scsc.csv", delimiter=",", skiprows=1)
            np.testing.assert_array_equal(csv_data[:, 0], result["time"])
            np.testing.assert_array_equal(csv_data[:, 1:5], result["runs"]["scsc"]["outputs"])
            np.testing.assert_array_equal(csv_data[:, 5:], result["runs"]["scsc"]["inputs"])
            with np.load(directory / "trajectories.npz") as saved:
                np.testing.assert_array_equal(saved["training_x"], result["training"]["inputs"])
                np.testing.assert_array_equal(saved["ccsc_measured"], result["runs"]["ccsc"]["measured"])
            self.assertTrue((directory / "metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
