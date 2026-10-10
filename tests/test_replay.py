"""Runtime contracts for platform simulation, artifacts and failure handling."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from platform_core.replay.config import validate_config
from platform_core.replay.simulation import simulate, SimulationCancelled
from platform_core.replay.runner import run, artifact_path, ARTIFACT_NAMES


class ReplayContractTests(unittest.TestCase):
    def test_live_snapshots_leave_solver_and_replay_history_unchanged(self):
        snapshots = []
        def observe(t, end, snapshot):
            if 5 < t < end:
                snapshots.append(snapshot())
        expected = simulate({"sim_time": 6.5})
        actual = simulate({"sim_time": 6.5}, on_snapshot=observe)
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertLess(snapshots[0]["t"][-1], actual["t"][-1])
        self.assertGreater(snapshots[0]["attack"][-1], 0)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        for snapshot in snapshots:
            for key, values in snapshot.items():
                np.testing.assert_allclose(values, actual[key][:len(values)], rtol=1e-12, atol=1e-12)

    def test_config_and_cancellation(self):
        for value in (0, 50.1, True, "50", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_config({"sim_time": value})
        with self.assertRaises(ValueError):
            validate_config({"delay": 10})
        with self.assertRaises(SimulationCancelled):
            simulate(cancelled=lambda: True)

    def test_artifacts_progress_and_failure(self):
        with TemporaryDirectory() as directory:
            progress = []
            result = run({"sim_time": .2}, run_dir=directory, progress=lambda p, m: progress.append(p))
            self.assertEqual(result["module"], "replay")
            self.assertEqual(len(result["plots"]), 4)
            self.assertEqual(progress[-1], 100)
            self.assertEqual(progress, sorted(progress))
            json.dumps(result, allow_nan=False)
            for name in ARTIFACT_NAMES:
                self.assertGreater(artifact_path(directory, name).stat().st_size, 20)
            for name in ("../signals.csv", "tracking.pdf", "replay_config.json"):
                with self.assertRaises(FileNotFoundError):
                    artifact_path(directory, name)
        with TemporaryDirectory() as directory, patch("platform_core.replay.runner.simulate", side_effect=RuntimeError("solver failed")):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                run(run_dir=directory)
            self.assertFalse((Path(directory) / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
