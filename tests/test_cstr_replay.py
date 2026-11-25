from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from platform_core.risk_control import replay
from platform_core.risk_control.presentation import tokens
from platform_core.risk_control.replay_display import diagram_sample, display
from platform_core.risk_control.replay_session import ReplayService


class ReplayTests(unittest.TestCase):
    def test_process_diagram_uses_branch_state_at_the_selected_boundary(self):
        trial = replay.Trial("constant")
        for _ in range(replay.START):
            trial.advance()
        states = {arm: branch.state.copy() for arm, branch in trial.branches.items()}
        row = trial.advance()
        for arm in replay.ARMS:
            with self.subTest(arm=arm):
                group = row["groups"][arm]
                sample = diagram_sample(row, arm, display(row, arm, True))
                self.assertEqual((sample["C"], sample["T"], sample["Tc"]), tuple(states[arm]))
                self.assertEqual(sample["Qc"], group["command_L_min"])
                labels = tokens(sample)
                self.assertEqual(labels["OBSBRANCH"], "0" if arm == "pi" else "1")
                self.assertEqual(labels["OBSACTIVE"], "1" if arm in ("raw", "guard", "candidate", "continuous") else "0")
                self.assertEqual((labels["MEASUREMARK"], labels["EXECMARK"]), ("0", "0"))
        guard = row["groups"]["guard"]
        self.assertNotEqual(guard["T_true_K"], guard["T_to_observer_K"])
        del guard["C_true_mol_L"]
        saved = diagram_sample(row, "guard", display(row, "guard", True))
        self.assertEqual(tokens(saved)["C"], "—")
        self.assertEqual(tokens(saved)["TEMP"], f"{guard['T_true_K']:.2f}")
        self.assertIsNone(diagram_sample(None, "guard", None))
        self.assertIsNone(diagram_sample({"groups": {"guard": {"valid": False}}}, "guard", None))

    def test_source_time_and_single_command_reproduce_rehearsal(self):
        for case, expected in (("constant",(.7630205133303491,.4360746616897586)),
                               ("removed",(.3157692074465615,1.2392377252584084))):
            trial = replay.Trial(case)
            rows = [trial.advance() for _ in range(91)]
            raw, guard = (rows[50]["groups"][k] for k in ("raw","guard"))
            self.assertEqual((raw["source_k"],guard["source_k"]),(10,10))
            self.assertEqual(raw["T_local_K"],guard["T_local_K"])
            self.assertNotEqual(raw["T_to_observer_K"],raw["T_local_K"])
            for point, command in ((raw,151.1523261424),(guard,149.798)):
                self.assertAlmostEqual(point["PI_L_min"]+point["comp_L_min"],point["command_L_min"])
                self.assertAlmostEqual(point["command_L_min"],command,places=3)
            self.assertEqual(rows[60]["groups"]["guard"]["status"],"reanchor")
            self.assertIsNone(rows[60]["groups"]["guard"]["innovation_K"])
            self.assertIsNotNone(rows[61]["groups"]["guard"]["innovation_K"])
            metrics = replay.scores(rows,trial.case)
            for name,target in zip(("raw","guard"),expected):
                self.assertAlmostEqual(metrics[name]["iae"],target,places=8)

    def test_dropouts_latency_and_unknown_timestamp(self):
        for case in ("long_gap","unknown","delay","local_missing"):
            sim = replay.Trial(case,ablation=True)
            rows = [sim.advance()["groups"] for _ in range(72)]
            if case == "local_missing":
                self.assertTrue(all(not g["valid"] and g["failed_at"]==50 for g in rows[-1].values()))
            else:
                self.assertEqual(rows[50]["guard"]["status"],"hold")
                self.assertEqual(rows[50]["immediate"]["weight"],0)
                if case == "long_gap":
                    self.assertEqual([rows[k]["guard"]["weight"] for k in (59,60,64,69,70,71)],[1.,.8,0.,0.,.2,.4])
                if case == "unknown":
                    self.assertIsNone(rows[50]["guard"]["source_k"])
                if case == "delay":
                    self.assertEqual(rows[50]["guard"]["source_k"],49)

    def test_local_reestimation_reproduces_withdrawal_and_constant_load(self):
        for name, expected_triggers, expected_iae in (
                ("removed", [55], (.3971369067097442, .2712742738329155)),
                ("constant", [], (.4360746616897586, .430887749791833))):
            with self.subTest(case=name):
                trial = replay.Trial(name)
                rows = [trial.advance() for _ in range(91)]
                points = [r["groups"]["candidate"] for r in rows]
                self.assertEqual([k for k, p in enumerate(points) if p["local_triggered"]], expected_triggers)
                for k in range(50, 60):
                    candidate, continuous = points[k], rows[k]["groups"]["continuous"]
                    self.assertEqual(candidate["source_k"], k - 40)
                    self.assertAlmostEqual(candidate["local_residual_K"],
                                           candidate["T_local_K"] - candidate["local_prediction_K"])
                    self.assertTrue(continuous["local_updating"])
                    self.assertEqual(continuous["status"], "local_update")
                    if not expected_triggers or k < expected_triggers[0]:
                        self.assertEqual(rows[k]["groups"]["guard"]["command_L_min"], candidate["command_L_min"])
                        self.assertFalse(candidate["local_updating"])
                    else:
                        self.assertTrue(candidate["local_updating"])
                if expected_triggers:
                    self.assertAlmostEqual(points[54]["local_residual_K"], -.06909, places=5)
                    self.assertAlmostEqual(points[55]["comp_L_min"], 1.2332, places=4)
                    self.assertLess(abs(points[56]["local_residual_K"]), .05)
                    self.assertTrue(points[56]["local_updating"])
                for arm in ("candidate", "continuous"):
                    first_current, next_current = (rows[k]["groups"][arm] for k in (60, 61))
                    self.assertEqual(first_current["status"], "reanchor")
                    self.assertIsNone(first_current["innovation_K"])
                    self.assertIsNone(first_current["local_prediction_K"])
                    self.assertFalse(first_current["local_updating"])
                    self.assertEqual(next_current["status"], "current")
                    self.assertIsNotNone(next_current["innovation_K"])
                metrics = replay.scores(rows, trial.case)
                for arm, expected in zip(("candidate", "continuous"), expected_iae):
                    self.assertAlmostEqual(metrics[arm]["iae"], expected, places=8)

    def test_local_trigger_state_clears_between_manual_segments(self):
        trial = replay.Trial("removed", manual=True)
        for _ in range(replay.START):
            trial.advance()
        trial.advance("begin")
        for _ in range(5):
            point = trial.advance()["groups"]["candidate"]
        self.assertTrue(point["local_triggered"])
        recovered = trial.advance("finish")["groups"]["candidate"]
        self.assertEqual(recovered["status"], "reanchor")
        self.assertEqual(recovered["local_streak"], 0)
        first = trial.advance("begin")["groups"]["candidate"]
        self.assertFalse(first["local_triggered"])
        self.assertFalse(first["local_updating"])
        self.assertLessEqual(abs(first["local_streak"]), 1)

    def test_archived_four_branch_run_keeps_its_original_groups(self):
        old_groups = {key: replay.ARMS[key] for key in ("raw", "guard", "fresh", "pi")}
        with TemporaryDirectory() as root:
            service = ReplayService(Path(root))
            try:
                with patch.dict(replay.ARMS, old_groups, clear=True):
                    session = service.create(auto_start=False)
                    with session.lock:
                        while session.last_published < 49:
                            session._tick()
                        old = session.run
                session.command("reset", "new", old)
                restored = session.frame(run=old, at=49, arm="candidate")
                self.assertEqual(set(restored["available_arms"]), set(old_groups))
                self.assertEqual(restored["arm"], "guard")
                self.assertEqual(restored["point"]["groups"].keys(), old_groups.keys())
                self.assertEqual(restored["display"], session.frame(run=old, at=49, arm="guard")["display"])
            finally:
                service.close()

    def test_idempotent_commands_history_and_cancel(self):
        with TemporaryDirectory() as root:
            service = ReplayService(Path(root))
            try:
                session = service.create({"speed":1200},auto_start=False)
                def until(predicate):
                    deadline = time.monotonic()+15
                    while time.monotonic()<deadline:
                        frame = session.frame()
                        if predicate(frame):
                            return frame
                        time.sleep(.015)
                    self.fail("replay clock timed out")
                initial = until(lambda f:f["live_seq"] == replay.START-1)
                old = initial["run"]
                self.assertFalse(initial["display"]["preview"])
                self.assertEqual(len(initial["samples"]),50)
                self.assertIsNone(initial["scored_through"])
                key = session.command("begin","queued",old)
                self.assertEqual(session.command("begin","queued",old),key)
                self.assertTrue(session.frame()["display"]["preview"])
                session.command("cancel","cancel",old)
                self.assertFalse(session.frame()["display"]["preview"])
                session.command("resume","go",old)
                until(lambda f:f["live_seq"]>95)
                session.command("pause","pause",old)
                historic = session.frame(old,50)
                self.assertIsNone(historic["band"])
                self.assertEqual(historic["point"]["groups"]["guard"]["source_k"],50)
                self.assertEqual(session.frame()["bands"], [])
                self.assertFalse(session.frame()["metrics"])
                reset = session.command("reset","new",old,{"scenario":"removed"})
                self.assertEqual(reset,session.command("reset","new",old,{"scenario":"removed"}))
                with self.assertRaises(ValueError):
                    session.command("cancel","bad",old)
                self.assertEqual(historic["point"],session.frame(old,50)["point"])
                self.assertNotEqual(old,session.frame()["live_run"])
            finally:
                service.close()

    def test_repeat_uses_current_boundary_and_preserves_history_and_scores(self):
        with TemporaryDirectory() as root:
            service = ReplayService(Path(root))
            try:
                session = service.create(auto_start=False)
                with session.lock:
                    while session.last_published < 73:
                        session._tick()
                    run = session.run
                    self.assertFalse(session.frame()["scheduled"])
                    session.command("begin","first",run)
                    session._tick()
                    first = session.frame()
                    self.assertEqual(first["bands"], [[7.4,7.5]])
                    self.assertEqual(first["point"]["groups"]["guard"]["source_k"],10)
                    observed = first["samples"][-1]
                    self.assertNotEqual(observed["local_Tc"], observed["received_Tc"])
                    self.assertNotEqual(observed["u_raw"], observed["u_guard"])
                    self.assertEqual(session.frame(arm="fresh")["samples"][-1]["received_Tc"],observed["local_Tc"])
                    self.assertIsNone(session.frame(arm="pi")["samples"][-1]["received_Tc"])
                    session.command("begin","repeat",run,{"scenario":"removed"})
                    session._tick()
                    gap = session.frame()
                    self.assertEqual((gap["point"]["groups"]["guard"]["source_k"], gap["pending"]), (75,"begin"))
                    self.assertEqual(gap["display"]["action"], "恢复首拍，对齐温度")
                    session._tick()
                    self.assertEqual(session.frame()["point"]["groups"]["guard"]["source_k"],10)
                    for _ in range(40):
                        session._tick()
                    final = session.frame()
                    self.assertEqual((final["segment"], final["scored_through"], final["scenario"]), (2,116,"removed"))
                    self.assertEqual(final["bands"], [[7.4,7.5],[7.6,8.6]])
                    self.assertEqual(final["metrics"]["guard"]["intervals"],40)
                    self.assertEqual(session.frame(at=79)["point"]["groups"]["guard"]["heat_true_K_min"],0)
                    self.assertEqual(session.frame(at=85)["point"]["groups"]["guard"]["source_k"],19)
                    self.assertEqual(session.frame(at=86)["point"]["groups"]["guard"]["status"],"reanchor")
                    session.command("begin","third",run,{"scenario":"constant"})
                    session._tick()
                    self.assertEqual(session.frame()["point"]["groups"]["guard"]["heat_true_K_min"],.6)
                    session.command("finish","early",run)
                    session._tick()
                    self.assertEqual(session.frame()["point"]["stop_seq"],118)
                    for _ in range(126):
                        session._tick()
                    session.command("reset","archive",run)
                self.assertEqual(session.frame(run, 74)["samples"], first["samples"])
                self.assertEqual(session.frame(run, 75)["metrics"]["guard"]["intervals"], 1)
                self.assertEqual(session.frame(run, 116)["metrics"], final["metrics"])
                self.assertEqual(session.frame(run, 116)["bands"],final["bands"])
                self.assertEqual(session.frame(run, 117)["metrics"]["guard"]["intervals"],0)
                self.assertEqual(session.frame(run, 240)["bands"],[])
            finally:
                service.close()

    def test_failed_journal_append_keeps_last_published_sample(self):
        with TemporaryDirectory() as root:
            service = ReplayService(Path(root))
            try:
                session = service.create(auto_start=False)
                with session.lock:
                    while session.last_published < replay.START-1:
                        session._tick()
                old = session.run
                with patch.object(session.log, "append", side_effect=OSError("journal unavailable")):
                    session.command("resume", "resume", old)
                    deadline = time.monotonic() + 5
                    while not session.frame()["error"] and time.monotonic() < deadline:
                        time.sleep(.01)
                frame = session.frame()
                self.assertEqual(frame["status"], "paused")
                self.assertEqual(frame["error"], "journal unavailable")
                self.assertEqual(frame["last"], frame["point"]["seq"])
                self.assertEqual(frame["live_seq"], replay.START-1)
                with self.assertRaises(ValueError):
                    session.frame(old, at=replay.START)
                session.command("reset", "replacement", old)
                self.assertEqual(session.frame(old)["point"], frame["point"])
            finally:
                service.close()
