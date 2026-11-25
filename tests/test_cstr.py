"""Plant, differentiable predictor and boundary semantics for the CSTR controllers."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from platform_core.risk_control import cstr, presentation
from platform_core.risk_control.dnn_mpc import dataset, model, validation
from platform_core.risk_control.dnn_mpc.mpc import Settings, neural_objective, solve
from platform_core.risk_control.session import Session
from platform_core.risk_control.session_lifecycle import lease_action


class SessionLeaseTests(unittest.TestCase):
    def test_disconnect_deadlines_respect_manual_pauses_and_connected_owners(self):
        cases = [
            (True, 0, 20, "running", None),
            (True, 0, 21, "running", "pause"),
            (True, 0, 120, "paused", None),
            (True, 0, 121, "paused", "expire"),
            (True, 1, 121, "running", None),
            (False, 0, 121, "running", None),
        ]
        for leased, connections, idle, status, expected in cases:
            with self.subTest(leased=leased, connections=connections, idle=idle, status=status):
                self.assertEqual(lease_action(leased, connections, idle, status), expected)


class PlantAndModelTests(unittest.TestCase):
    def test_equilibrium_step_direction_and_rk4_refinement(self):
        x, u = cstr.NOMINAL, cstr.NOMINAL_FLOW
        np.testing.assert_allclose(cstr.derivative(x, u, cstr.INLET), 0, atol=1e-9)
        np.testing.assert_allclose(cstr.step(x, u), x, atol=1e-9)
        self.assertLess(cstr.step(x, u+15)[1], x[1])
        self.assertGreater(cstr.step(x, u-15)[1], x[1])
        coarse, half, fine = (cstr.step(x, u+15, substep=h) for h in (.01, .005, .0025))
        self.assertLess(np.linalg.norm(half-fine), np.linalg.norm(coarse-half)/8)

    def test_model_contract_and_horizon_gradient(self):
        net = model.load()
        self.assertTrue(net.metadata["acceptance"]["passed"])
        z = np.array([.1, -.3, .25, .2, -.1])
        args = (net, cstr.NOMINAL, cstr.NOMINAL_FLOW, cstr.INLET, 432, .15)
        _, gradient = neural_objective(z, *args)
        eps = 1e-5
        finite = [(neural_objective(z+d, *args)[0]-neural_objective(z-d, *args)[0])/(2*eps)
                  for d in eps*np.eye(len(z))]
        np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=2e-7)
        with tempfile.TemporaryDirectory() as tmp, np.load(model.DEFAULT_FILE, allow_pickle=False) as saved:
            data = dict(saved)
            contract = json.loads(str(data["metadata"]))
            contract["inputs_units"][1] = "degC"
            path = Path(tmp)/"wrong.npz"
            np.savez(path, **{**data, "metadata": json.dumps(contract)})
            with self.assertRaises(ValueError):
                model.load(path)

    def test_gate_rejects_a_biased_candidate(self):
        net = model.load()
        for parameter in net.parameters():
            parameter.zero_()
        net.ym[1] += 4
        report = validation.check(net, dataset.generate(6, 12, 73), Settings(horizon=4))
        self.assertFalse(report["passed"])
        self.assertTrue(any("one_rmse" in reason for reason in report["reasons"]))

    def test_training_trajectories_and_temperature_scale(self):
        batch = dataset.generate(10, 7, 101)
        train, valid, test = dataset.split(batch)
        self.assertEqual([len(v.features) for v in (train, valid, test)], [7, 1, 2])
        np.testing.assert_array_equal(batch.features[:, 1:, :3], batch.targets[:, :-1])
        sample = dict(T=424, Tc=415, Qc=148, C=.1, T_ref=431, Ci=1, Ti=350, Tci=350)
        cool = presentation.tokens(sample)
        hot = presentation.tokens({**sample, "T": 438})
        self.assertNotEqual(cool["COLOR"], hot["COLOR"])
        self.assertIn(cool["COLOR"], presentation.render(sample))
        self.assertNotIn("@@", presentation.render(sample))


class DisturbanceTests(unittest.TestCase):
    def test_all_presets_restore_after_requested_duration_and_ten_recovery_intervals(self):
        from platform_core.risk_control.disturbance import InletTrial, options
        for preset in options()["presets"]:
            with self.subTest(preset=preset["id"]):
                trial = InletTrial("key", preset["id"], 1)
                self.assertEqual(trial.boundary(5, cstr.NOMINAL, 150., 430.), "disturbance_applied")
                self.assertEqual(np.count_nonzero(trial.input != cstr.INLET), 1)
                self.assertIn(preset["target"], trial.input)
                held = trial.compare(trial.input)
                self.assertFalse(trial.commit(6, 430., 430., held))
                self.assertEqual(trial.boundary(6, cstr.NOMINAL, 151., 430.), "disturbance_restored")
                for seq in range(7, 17):
                    self.assertEqual(trial.commit(seq, 430., 430., trial.compare(cstr.INLET)), seq == 16)
                self.assertEqual(trial.summary(16)["phase"], "complete")
                self.assertEqual(trial.held_flow, 150.)

    def test_separate_trials_have_no_connecting_line_or_shaded_recovery(self):
        rows = [dict(seq=i, t=i*.1, T=432.) for i in range(6)]
        for i, key, phase in ((1, "a", "active"), (2, "a", "recovery"), (4, "b", "active")):
            rows[i].update(trial_id=key, disturbance_phase=phase, T_hold=433., Qc_hold=149.)
        bands, groups = presentation.trial_traces(rows, .1)
        np.testing.assert_allclose(bands, [(0., .1), (.3, .4)])
        self.assertEqual([[r[0] for r in group] for group in groups], [[rows[i]["t"] for i in (0, 1, 2)], [rows[i]["t"] for i in (3, 4)]])


class CsvTrajectoryTests(unittest.TestCase):
    def test_csv_pairs_within_each_run_and_splits_whole_trajectories(self):
        from platform_core.risk_control.dnn_mpc.dataset import example_csv, read_csv, split
        lines = example_csv().splitlines()
        # Accept interleaved or unsorted rows and distinct trajectory lengths.
        data = read_csv(("\n".join([lines[0], *reversed(lines[1:-1])])).encode("utf-8-sig"))
        self.assertEqual(sorted(len(run) for run in data.runs), [8, 9, 9, 9])
        partitions = split(data)
        self.assertEqual([len(part) for part in partitions], [2, 1, 1])
        for part in partitions:
            x, y = part.flat()
            np.testing.assert_allclose(y[0], part.runs[0][1, :3])
            self.assertEqual(len(x), sum(len(run)-1 for run in part.runs))
        for invalid in (lines[:2], [*lines, lines[-1]], lines[:3]+lines[4:],
                        [lines[0], lines[1].replace(",0,", ",x,", 1), *lines[2:]]):
            with self.assertRaises(ValueError):
                read_csv("\n".join(invalid).encode())


class SessionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.s = Session(Path(self.temp.name), model.load(), Settings(horizon=4, speed=10))
        self.addCleanup(self.s.close)
        self.old = self.s.model.version
        self.candidate = model.load()
        self.candidate.metadata = {**self.candidate.metadata, "version": "candidate"}
        # Completed ideal transitions at the equilibrium form one continuous run.
        x = cstr.NOMINAL.copy()
        for _ in range(24):
            y = cstr.step(x, cstr.NOMINAL_FLOW)
            self.s.history.append((np.r_[x, cstr.NOMINAL_FLOW, cstr.INLET], y))
            x = y

    def wait(self, predicate):
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            value = self.s.snapshot()
            if predicate(value):
                return value
            time.sleep(.01)
        self.fail(str(self.s.snapshot()))

    def test_pulse_boundaries_match_the_controller_plant_comparison_and_training_inputs(self):
        key, preset = "pulse", "feed_concentration"
        request = self.s.command("disturbance", key, preset=preset, duration_cycles=2)
        self.assertEqual(self.s.command("disturbance", key, preset=preset, duration_cycles=2), request)
        with self.assertRaisesRegex(ValueError, "正在进行"):
            self.s.command("disturbance", "second", preset=preset)
        self.assertEqual(self.s.snapshot()["disturbance"]["phase"], "queued")
        observed = []

        def record(*args, **kwargs):
            observed.append(kwargs["inlet"].copy())
            return solve(*args, **kwargs)

        with patch("platform_core.risk_control.session.solve", side_effect=record):
            self.s.command("start", "go")
            self.wait(lambda v: v["seq"] >= 3)
            self.s.command("pause", "pause")
        snap = self.s.snapshot()
        rows = snap["samples"][:4]
        self.assertEqual([r["Ci"] for r in rows], [1, 1.03, 1.03, 1])
        self.assertEqual([r.get("disturbance_phase") for r in rows], [None, "active", "active", "recovery"])
        held = cstr.NOMINAL.copy()
        for i, (previous, r) in enumerate(zip(rows, rows[1:])):
            inlet = [r[k] for k in ("Ci", "Ti", "Tci")]
            state = [previous[k] for k in ("C", "T", "Tc")]
            actual = [r[k] for k in ("C", "T", "Tc")]
            np.testing.assert_allclose(observed[i], inlet)
            np.testing.assert_allclose(actual, cstr.step(state, r["Qc"], inlet))
            features, result = self.s.history[24+i]
            np.testing.assert_allclose(features, np.r_[state, r["Qc"], inlet])
            np.testing.assert_allclose(result, actual)
            held = cstr.step(held, rows[0]["Qc"], inlet)
            self.assertAlmostEqual(r["T_hold"], held[1])
            self.assertEqual(r["Qc_hold"], rows[0]["Qc"])
        marks = [(e["event"], e["first_interval"]) for e in snap["events"] if "first_interval" in e]
        self.assertEqual(marks, [("disturbance_applied", 1), ("disturbance_restored", 3)])
        self.assertEqual(self.s.sample_window(self.s.run, 1, 3), rows[1:])
        bands, groups = presentation.trial_traces(rows, cstr.SAMPLE_MIN)
        self.assertEqual(bands, [[0, .2]])
        self.assertEqual(len(groups[0]), 4)
        self.assertIn("恒流对照", presentation.trial_status(snap["disturbance"])["metrics"])

    def test_manual_clear_mid_cycle_preserves_in_flight_sample_and_reset_clears_trial(self):
        entered, release = threading.Event(), threading.Event()

        def gated(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                release.wait(3)
            return solve(*args, **kwargs)

        self.s.command("disturbance", "warm", preset="feed_temperature")
        with patch("platform_core.risk_control.session.solve", side_effect=gated):
            self.s.command("start", "start")
            try:
                self.assertTrue(entered.wait(4))
                self.s.command("clear_disturbance", "withdraw")
            finally:
                release.set()
            snap = self.wait(lambda v: v["seq"] >= 2)
            self.s.command("pause", "pause")
        self.assertEqual([r["Ti"] for r in snap["samples"][:3]], [350, 352, 350])
        self.assertEqual(snap["disturbance"]["restore_min"], cstr.SAMPLE_MIN)
        self.s.command("reset", "new")
        self.assertIsNone(self.s.snapshot()["disturbance"])

    def test_rejected_cancelled_and_obsolete_updates_preserve_active_model(self):
        info, data = self.s.prepare_update()
        self.assertEqual(data[1].features.shape, (1, 24, 7))
        self.assertIs(self.s.prepare_update()[0], info)
        self.assertFalse(self.s.finish_update(info, None, {"passed": False}))
        info, _ = self.s.prepare_update()
        info["cancel"].set()
        self.assertFalse(self.s.finish_update(info, self.candidate, {}))
        info, _ = self.s.prepare_update()
        old_run = self.s.run
        answer = self.s.command("reset", "reset-once")
        self.assertEqual(self.s.command("reset", "reset-once"), answer)
        self.assertNotEqual(self.s.run, old_run)
        self.assertFalse(self.s.finish_update(info, self.candidate, {}))
        self.assertEqual(self.s.model.version, self.old)
        with self.assertRaisesRegex(ValueError, "24"):
            self.s.prepare_update()

    def test_candidate_accepted_mid_solve_activates_at_next_boundary(self):
        entered, release = threading.Event(), threading.Event()

        def controlled_solve(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                release.wait(2)
            return solve(*args, **kwargs)
        info, _ = self.s.prepare_update()
        with patch("platform_core.risk_control.session.solve", side_effect=controlled_solve):
            self.s.command("start", "one")
            self.assertTrue(entered.wait(5))
            self.assertTrue(self.s.finish_update(info, self.candidate, {"passed": True}))
            self.s.command("reference", "setpoint", value=432)
            release.set()
            self.wait(lambda v: v["seq"] >= 2)
            self.s.command("pause", "hold")
        snapshot = self.s.snapshot()
        self.assertEqual([v["model"] for v in snapshot["samples"][:3]], [self.old, self.old, "candidate"])
        self.assertEqual(snapshot["samples"][2]["T_ref"], 432)
        event = next(e for e in snapshot["events"] if e["event"] == "activated")
        self.assertEqual(event["first_interval"], 2)
        self.assertEqual(self.s.export()["seq"], snapshot["seq"])

    def test_failed_first_solve_rolls_back_without_double_integration(self):
        info, _ = self.s.prepare_update()
        self.s.command("pause", "hold")
        self.assertTrue(self.s.finish_update(info, self.candidate, {}))
        self.assertEqual(self.s.model.version, self.old)

        def first_fails(net, *args, **kwargs):
            if net.version == "candidate":
                raise RuntimeError("candidate solve failed")
            return solve(net, *args, **kwargs)
        with patch("platform_core.risk_control.session.solve", side_effect=first_fails):
            self.s.command("resume", "go")
            self.wait(lambda v: v["seq"] >= 1)
            self.s.command("end", "end")
        v = self.s.snapshot()
        np.testing.assert_allclose([v["samples"][1][key] for key in ("C", "T", "Tc")],
                                   cstr.step(cstr.NOMINAL, v["samples"][1]["Qc"]))
        self.assertEqual(v["samples"][1]["attempts"], ["candidate", self.old])
        self.assertEqual(v["samples"][1]["model"], self.old)
        self.assertIn("rollback", [e["event"] for e in v["events"]])

    def test_overdue_candidate_uses_pi_and_late_success_cannot_overwrite_or_queue(self):
        info, _ = self.s.prepare_update()
        self.s.finish_update(info, self.candidate, {"passed": True})
        entered, release = threading.Event(), threading.Event()
        calls = []

        def sometimes_late(net, state, previous, ref, settings, *args, **kwargs):
            calls.append(net.version)
            if len(calls) == 1:
                entered.set()
                release.wait(8)
                flow = settings.flow_max
            else:
                flow = previous
            return np.full(settings.horizon, flow), {"solve_ms": 0.5}

        with patch("platform_core.risk_control.session.solve", side_effect=sometimes_late):
            self.s.command("start", "start")
            try:
                self.assertTrue(entered.wait(4))
                snap = self.wait(lambda v: v["seq"] >= 2)
                self.assertEqual(snap["status"], "running")
                self.assertEqual(calls, ["candidate"])
                self.assertEqual(snap["model"], self.old)
                self.assertEqual([r["mpc_outcome"] for r in snap["samples"][1:3]], ["timeout", "busy"])
                self.assertTrue(presentation.control_status(snap)["active"])
                for i in (1, 2):
                    row = snap["samples"][i]
                    self.assertEqual(row["control_source"], "pi")
                    self.assertIsNone(row["control_model"])
                    self.assertIsNone(row["solve_ms"])
                    self.assertEqual(row["fallback_streak"], i)
                    self.assertLess(row["decision_budget_ms"], 600)
                    prev = snap["samples"][i-1]
                    expected = cstr.step([prev[k] for k in ("C", "T", "Tc")], row["Qc"])
                    np.testing.assert_allclose(expected, [row[k] for k in ("C", "T", "Tc")])
                    self.assertAlmostEqual(self.s.history[23+i][0][3], row["Qc"])
            finally:
                release.set()
            snap = self.wait(lambda v: v["latest"].get("control_source") == "mpc")
            self.s.command("pause", "preserve")
        self.assertEqual(snap["latest"]["control_model"], self.old)
        self.assertLess(snap["latest"]["Qc"], self.s.settings.flow_max)
        self.assertIn("mpc_recovered", [e["event"] for e in snap["events"]])
        self.assertIn("MPC 已恢复", presentation.control_status(snap)["message"])
        timing = self.s.export()["timing"]
        self.assertGreaterEqual(timing["deadline_misses"], 2)
        self.assertEqual(timing["failed"], 0)
        self.assertEqual(timing["deadline_misses"], timing["fallback_cycles"])

    def test_reset_during_solve_discards_reply_and_resets_pi_alarm(self):
        entered, release = threading.Event(), threading.Event()
        def late(net, x, previous, ref, settings, *args, **kwargs):
            entered.set()
            release.wait(8)
            return np.full(settings.horizon, settings.flow_max), {"solve_ms": 0.5}
        with patch("platform_core.risk_control.session.solve", side_effect=late):
            self.s.command("start", "go")
            try:
                self.assertTrue(entered.wait(4))
                self.wait(lambda v: v["latest"].get("control_source") == "pi")
                self.s.command("reset", "reset")
                self.assertIsNone(self.s.snapshot()["last_fallback"])
            finally:
                release.set()
            with self.s.lock:
                self.assertTrue(self.s.lock.wait_for(lambda: self.s.solver.job is None, timeout=3))
        self.assertEqual(self.s.snapshot()["seq"], 0)
        self.assertEqual(len(self.s.history), 0)
        self.assertEqual(self.s.previous, cstr.NOMINAL_FLOW)

    def test_invalid_mpc_and_large_reentry_hold_pi_until_a_bounded_proposal(self):
        calls = []
        def propose(net, x, previous, ref, settings, *args, **kwargs):
            calls.append(previous)
            flow = [np.nan, settings.flow_max, previous+1][min(len(calls)-1, 2)]
            return np.full(settings.horizon, flow), {"solve_ms": 0.2}
        with patch("platform_core.risk_control.session.solve", side_effect=propose):
            self.s.command("start", "go")
            snap = self.wait(lambda v: v["seq"] >= 3)
            self.s.command("pause", "stop")
        rows = snap["samples"][1:4]
        self.assertEqual([r["mpc_outcome"] for r in rows], ["failed", "transfer_limit", "ok"])
        self.assertEqual([r["fallback_streak"] for r in rows], [1, 2, 0])
        self.assertAlmostEqual(rows[2]["Qc"], rows[1]["Qc"]+1)
        self.assertEqual(self.s.export()["timing"]["deadline_misses"], 0)

    def test_candidate_outside_transfer_band_rolls_back_and_retries_previous_model(self):
        info, _ = self.s.prepare_update()
        calls = []
        def sequence(net, state, previous, ref, settings, *args, **kwargs):
            calls.append(net.version)
            if len(calls) == 1:
                raise RuntimeError("first solve unavailable")
            flow = settings.flow_max if net.version == "candidate" else previous
            return np.full(settings.horizon, flow), {"solve_ms": 1.}
        with patch("platform_core.risk_control.session.solve", side_effect=sequence):
            self.s.command("start", "start")
            # Deliver the candidate after PI has taken control, at a known boundary.
            self.wait(lambda v: v["latest"].get("control_source") == "pi")
            self.s.command("pause", "hold")
            self.assertTrue(self.s.finish_update(info, self.candidate, {"passed": True}))
            self.s.command("resume", "go")
            self.wait(lambda v: any(e["event"] == "mpc_recovered" for e in v["events"]))
            self.s.command("end", "stop")
        samples = self.s.snapshot()["samples"]
        recovered = next(row for row in samples if row.get("attempts") == ["candidate", self.old])
        self.assertEqual((recovered["control_model"], recovered["control_source"]), (self.old, "mpc"))
        self.assertIsNone(self.s.trial_previous)

    def test_backup_pi_limits_and_antiwindup(self):
        from platform_core.risk_control.dnn_mpc.supervision import backup_flow
        cfg = self.s.settings
        q, integral = backup_flow(150., 5., 150., cfg)
        self.assertEqual((q, integral), (155., 150.))
        self.assertEqual(backup_flow(100., -2., 102., cfg), (100., 100.))
        q, integral = backup_flow(150., 0.2, 151., cfg)
        self.assertAlmostEqual(q, 152.2)
        self.assertAlmostEqual(integral, 150.1)

    def test_background_owner_keeps_running_and_manual_model_switch_keeps_run(self):
        with self.s.lock:
            self.s.leased = True
            self.s.owner_connected()
            self.s.last_seen = time.monotonic()-150
            self.candidate.metadata["acceptance"] = {"passed": True}
            self.s.command("model", "choice", reset_model=self.candidate)
            initial_run = self.s.run
            self.s.command("start", "background")
            self.s.last_seen = time.monotonic()-150
        snap = self.wait(lambda s: s["seq"] >= 1)
        self.assertEqual(snap["run"], initial_run)
        self.assertEqual(snap["samples"][1]["model"], "candidate")
        self.assertEqual(snap["status"], "running")
        self.s.owner_disconnected()
        self.s.owner_disconnected()
        self.assertEqual(self.s.owner_connections, 0)

    def test_history_reads_samples_before_the_ring_and_rejects_old_run(self):
        with self.s.lock:
            for seq in range(1, 601):
                self.s._sample(150., 150., self.old, {}, 0., self.s.state, seq)
                self.s.seq = seq
        live = self.s.snapshot()
        self.assertEqual([row["seq"] for row in live["samples"]], list(range(120, 601)))
        self.assertEqual(self.s.sample_window(self.s.run, 120, 481), live["samples"])
        for first in (0, 119, 128, 475, 595):
            rows = self.s.sample_window(self.s.run, first, 121)
            self.assertEqual([r["seq"] for r in rows], list(range(first, min(601, first+121))))
        previous = self.s.run
        self.s.command("reset", "new")
        with self.assertRaises(ValueError):
            self.s.sample_window(previous, 0, 10)

    def test_owner_lease_pauses_and_expires_without_observer_refresh(self):
        with self.s.lock:
            self.s.leased = True
            self.s.status = "running"
            self.s.last_seen = time.monotonic()-25
            self.s.snapshot(owner="observer")
            self.s.lock.notify_all()
        self.wait(lambda v: v["status"] == "paused")
        with self.s.lock:
            self.s.last_seen = time.monotonic()-125
            self.s.lock.notify_all()
        self.s.worker.join(4)
        self.assertTrue(self.s.closed)

    def test_failed_sample_record_does_not_advance_plant_or_training_history(self):
        original = self.s.log.append
        def no_sample(kind, record):
            if kind == "sample":
                raise OSError("storage failure")
            original(kind, record)
        with patch.object(self.s.log, "append", side_effect=no_sample):
            self.s.command("start", "run")
            state = self.wait(lambda v: v["status"] == "paused")
        self.assertEqual(state["seq"], 0)
        self.assertEqual(state["latest"]["seq"], 0)
        self.assertEqual(len(self.s.history), 24)
        np.testing.assert_array_equal(self.s.state, cstr.NOMINAL)
        self.assertEqual(self.s.export()["seq"], 0)
