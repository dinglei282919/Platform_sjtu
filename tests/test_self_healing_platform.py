"""Platform contracts, beyond the copied algorithm/plant regression tests."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from platform_core.risk_control.self_healing.config import catalog, defaults, validate_config
from platform_core.risk_control.self_healing.runner import ARTIFACT_NAMES, artifact_path, run
from platform_core.risk_control.self_healing.simulation import ExperimentConfig, SimulationCancelled, run_experiment


def quick_config(scenario="water"):
    return {**defaults(scenario), "duration": 1., "dt": .1, "batch_duration": .1,
            "fault_start": .3, "training_batches": 8, "training_hold": .2}


class PlatformSelfHealingTests(unittest.TestCase):
    def test_shared_presets_and_strict_resource_validation(self):
        for scenario, config in catalog()["defaults"].items():
            self.assertEqual(config, validate_config({"scenario": scenario}))
        for change in ({"duration": True}, {"seed": 1.5}, {"dt": "0.1"}, {"dt": float("nan")},
                       {"duration": 10**400}, {"seed": -1}, {"training_batches": 7},
                       {"duration": 5000, "dt": .0001}, {"output_dir": "elsewhere"},
                       {"scenario": []}, {"ccsc_centering": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(change)

    def test_progress_and_cancellation_do_not_change_numerics(self):
        for scenario in catalog()["scenarios"]:
            config = ExperimentConfig(**quick_config(scenario))
            expected = run_experiment(config)
            progress = []
            actual = run_experiment(config, cancelled=lambda: False, on_progress=lambda p, m: progress.append(p))
            self.assertEqual(progress, sorted(progress))
            for method in expected["runs"]:
                for key in ("outputs", "inputs", "measured"):
                    np.testing.assert_array_equal(expected["runs"][method][key], actual["runs"][method][key])
            count = 0
            def stop():
                nonlocal count
                count += 1
                return count >= 5
            with self.assertRaises(SimulationCancelled):
                run_experiment(config, cancelled=stop)
            # Stop after identification too: evaluation steps must remain interruptible.
            progress.clear()
            with self.assertRaises(SimulationCancelled):
                run_experiment(config, cancelled=lambda: bool(progress and progress[-1] >= 20),
                               on_progress=lambda p, m: progress.append(p))

    def test_complete_artifacts_failure_retry_and_existing_directory_protection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            progress = []
            result = run(quick_config(), run_dir=root / "success", progress=lambda p, m: progress.append(p))
            self.assertEqual(result["module"], "self_healing")
            self.assertEqual(result["config"], asdict(ExperimentConfig(**quick_config())))
            self.assertEqual(progress[-1], 100)
            self.assertEqual(progress, sorted(progress))
            json.dumps(result, allow_nan=False)
            for name in ARTIFACT_NAMES:
                self.assertGreater(artifact_path(root / "success", name).stat().st_size, 20)
            for name in ("../summary.json", "progress.json", "self_healing_config.json", "outputs.pdf"):
                with self.assertRaises(FileNotFoundError):
                    artifact_path(root / "success", name)
            before = (root / "success" / "result.json").read_bytes()
            with self.assertRaises(ValueError):
                run(quick_config(), run_dir=root / "success")
            self.assertEqual(before, (root / "success" / "result.json").read_bytes())
            with patch("platform_core.risk_control.self_healing.runner.run_experiment", side_effect=RuntimeError("solver failed")):
                with self.assertRaisesRegex(RuntimeError, "solver failed"):
                    run(quick_config(), run_dir=root / "failure")
            self.assertFalse((root / "failure" / "result.json").exists())
            self.assertIn("solver failed", (root / "failure" / "progress.json").read_text(encoding="utf-8"))
            with self.assertRaises(SimulationCancelled):
                run(quick_config(), run_dir=root / "cancelled", cancelled=lambda: True)
            self.assertFalse((root / "cancelled" / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
