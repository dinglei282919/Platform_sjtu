"""Runtime contracts for platform simulation, artifacts and failure handling."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch
import numpy as np
from platform_core.fdi.config import validate_config
from platform_core.fdi.simulation import simulate, SimulationCancelled
from platform_core.fdi.runner import run, artifact_path, ARTIFACT_NAMES
from platform_core.fdi import runner


class FdiContractTests(unittest.TestCase):
    def test_live_snapshots_leave_solver_outputs_unchanged(self):
        snapshots = []
        def observe(t, end, snapshot):
            if not snapshots and 0 < t < end:
                snapshots.append(snapshot())
        expected = simulate({"sim_time": .2})
        actual = simulate({"sim_time": .2}, on_snapshot=observe)
        self.assertTrue(snapshots)
        self.assertLess(snapshots[0]["t"][-1], actual["t"][-1])
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        for key, values in snapshots[0].items():
            np.testing.assert_array_equal(values, actual[key][:len(values)])

    def test_invalid_configs_and_cancellation(self):
        for value in (0, 10.1, float("nan"), float("inf"), True, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"sim_time": value})
        with self.assertRaises(ValueError):
            validate_config({"attack_scale": 5})
        with self.assertRaises(SimulationCancelled):
            simulate(cancelled=lambda: True)

    def test_run_exports_all_four_groups_and_finite_json(self):
        with TemporaryDirectory() as folder:
            result = run({"sim_time": .2}, run_dir=folder)
            self.assertEqual(len(result["plots"]), 4)
            for name in ARTIFACT_NAMES:
                path = artifact_path(folder, name)
                self.assertGreater(path.stat().st_size, 20)
            self.assertEqual(json.loads((Path(folder) / "progress.json").read_text(encoding="utf-8"))["percent"], 100)
            with np.load(Path(folder) / "signals.npz") as signals:
                self.assertTrue(np.isposinf(signals["DF"][0]))
            json.dumps(result, allow_nan=False)
            for name in ("../signals.npz", "missing.svg", "fdi_config.json"):
                with self.assertRaises(FileNotFoundError):
                    artifact_path(folder, name)

    def test_failed_simulation_does_not_publish_success(self):
        with TemporaryDirectory() as folder:
            with patch("platform_core.fdi.runner.simulate", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    run(run_dir=folder)
            self.assertFalse((Path(folder) / "result.json").exists())

    @unittest.skipUnless(os.name == "nt", "Windows reader/rename sharing semantics")
    def test_progress_writer_recovers_when_active_reader_closes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            runner.write_json(path, {"percent": 10})
            blocked, failures = Event(), []
            replace = os.replace

            def observed_replace(source, destination):
                try:
                    return replace(source, destination)
                except OSError:
                    blocked.set()
                    raise

            def write():
                try:
                    runner.write_json(path, {"percent": 20})
                except Exception as exc:
                    failures.append(exc)

            with patch.object(runner.os, "replace", side_effect=observed_replace):
                with path.open("rb") as reader:
                    worker = Thread(target=write)
                    worker.start()
                    try:
                        self.assertTrue(blocked.wait(5), "No real Windows sharing conflict observed")
                        self.assertEqual(json.loads(reader.read())["percent"], 10)
                    finally:
                        reader.close()
                        worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(json.loads(path.read_text())["percent"], 20)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_progress_retries_are_bounded_and_preserve_previous_file(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            runner.write_json(path, {"percent": 10})
            for code in (5, 32, 33, 112):
                error = OSError("test write conflict")
                error.winerror = code
                with self.subTest(code=code), patch.object(runner.os, "replace", side_effect=error) as replace, \
                        patch.object(runner.time, "sleep") as sleep:
                    with self.assertRaises(OSError) as raised:
                        runner.write_json(path, {"percent": 20})
                self.assertIs(raised.exception, error)
                self.assertEqual(replace.call_count, 1 if code == 112 else 11)
                self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1.6)
                self.assertEqual(json.loads(path.read_text())["percent"], 10)
                self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_progress_error_does_not_mask_simulation_failure(self):
        original = RuntimeError("test solver failure")
        write_json = runner.write_json
        logs = []

        def write_status(path, value):
            if value.get("message") == str(original):
                raise PermissionError("test failure progress write")
            return write_json(path, value)

        with TemporaryDirectory() as folder, patch.object(runner, "write_json", side_effect=write_status), \
                patch.object(runner, "simulate", side_effect=original):
            with self.assertRaises(RuntimeError) as raised:
                run(run_dir=folder, log=logs.append)
        self.assertIs(raised.exception, original)
        self.assertTrue(any("无法写入失败进度" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
