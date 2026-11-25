"""Module-3 numerical and lifecycle regressions against the frozen CSTR trials."""
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from uuid import uuid4

from platform_core.risk_control.fdi import BOOK, RECOVERY_STEPS, Comparison, Config
from platform_core.risk_control.fdi_display import DEMO_VIEWS, view
from platform_core.risk_control.fdi_metrics import Scores
from platform_core.risk_control.fdi_session import FdiService


def wait_for(session, predicate):
    deadline = time.monotonic()+90
    while time.monotonic() < deadline:
        snap = session.snapshot()
        if snap["error"]:
            raise AssertionError(snap["error"])
        if predicate(snap):
            return snap
        time.sleep(.008)
    raise AssertionError("CSTR sample deadline exceeded")


class FdiTests(unittest.TestCase):
    def test_demo_presets_and_paired_plot_signals(self):
        for side, presets, baseline, compensated, observed in (
            ("measurement", ["readout"], "raw", "guard", ["Tm", "T", "witness"]),
            ("execution", ["delivery", "mixed"], "bare", "full", ["Tcm", "Tc", "prior_dnn_Tc"]),
        ):
            with self.subTest(side=side):
                demo = DEMO_VIEWS[side]
                self.assertEqual(list(demo["presets"]), presets)
                self.assertEqual(demo["protected_group"], compensated)
                self.assertEqual([chart["unit"] for chart in demo["charts"]], ["K", "L/min", "K"])
                expected = [
                    [(compensated, key) for key in observed],
                    [(baseline, "Qc_cmd"), (compensated, "Qc_cmd")],
                    [(baseline, "T"), (compensated, "T"), (compensated, "T_ref")],
                ]
                self.assertEqual([[(s["group"], s["key"]) for s in chart["signals"]]
                                  for chart in demo["charts"]], expected)
                for preset in presets:
                    engine = Comparison(Config.parse(dict(side=side, scenario=preset)))
                    first, _ = engine.advance(None)
                    current, _ = engine.advance("begin")
                    for series in expected:
                        for group, key in series:
                            self.assertIsInstance(current["groups"][group][key], float)
                    label = demo["presets"][preset]
                    note = demo["preset_notes"][preset]
                    if side == "measurement":
                        offset = BOOK["cases"][preset]["readout_K"]
                        self.assertIn(f"偏高 {offset:.2f} K", label)
                        self.assertIn(f"高 {offset:.2f} K", note)
                        self.assertIn("430.88 K", note)
                        self.assertIn(f"{430.88 + offset:.2f} K", note)
                    else:
                        sample = current["groups"][compensated]
                        shortfall = sample["Qc_cmd"] - sample["Qc"]
                        self.assertIn(f"少 {shortfall:.1f} L/min", label)
                        if sample["load"]:
                            heat = f"{sample['load']:+.2f} K/min"
                            self.assertIn(heat, label)
                            self.assertIn(heat, note)
                        else:
                            self.assertIn("0 K/min", note)
                            self.assertIn(f"{150.0 - shortfall:.1f} L/min", note)
                        self.assertIsNone(first["groups"][compensated]["prior_dnn_Tc"])
                        self.assertEqual(current["groups"][compensated]["prior_dnn_Tc"],
                                         first["groups"][compensated]["dnn_Tc"])

    def test_primary_traces_and_causal_response(self):
        # IAE in K·min from the archived seed-1301 runs, with the sample at [50,100).
        expected = {"measurement": {"raw": 3.2217505114544114, "guard": .23862178826908576, "fixed": .14001543543656111},
                    "execution": {"pi": 2.5621929876195626, "full": .34434796537224205,
                                  "bare": 2.01296488717756, "outer": 1.988194756288027, "inner": .3405177003921949}}
        for side, arms in expected.items():
            engine = Comparison(Config.parse({"side":side}))
            scores = {arm:Scores() for arm in arms}
            previous = None
            for k in range(181):
                row, events = engine.advance("begin" if k == 50 else None)
                for name, sample in row["groups"].items():
                    self.assertTrue(sample["valid"])
                    scores[name].record(sample, row, engine.stop)
                    if side == "execution":
                        self.assertNotIn("flow_read_previous", sample)
                if side == "execution" and 0 < k < 180:
                    full = row["groups"]["full"]
                    self.assertAlmostEqual(full["jacket_innovation"], full["Tcm"]-previous["dnn_Tc"])
                if side == "execution":
                    previous = row["groups"]["full"]
                if k == 50:
                    self.assertEqual([e["event"] for e in events], ["segment_start"])
                if k == 51:
                    self.assertEqual(events[0]["event"], "feedback_source" if side == "measurement" else "compensation_response")
            self.assertTrue(row["terminal"])
            for name, value in arms.items():
                self.assertAlmostEqual(scores[name].report()["iae"], value, places=9)
                self.assertIsNone(row["groups"][name]["Qc"])

    def test_missing_jacket_keeps_only_valid_feedback_advancing(self):
        for side, base, guarded in (("measurement","raw","guard"), ("execution","pi","full")):
            engine = Comparison(Config.parse(dict(side=side,scenario="missing_jacket")))
            for k in range(9):
                row, _ = engine.advance("begin" if k == 1 else None)
            affected = row["groups"][guarded]
            self.assertEqual((affected["seq"], affected["failure"]["seq"]), (4,5))
            self.assertFalse(affected["valid"])
            self.assertTrue(row["groups"][base]["valid"])
            self.assertEqual(row["groups"][base]["seq"], 8)

    def test_two_complete_injections_separate_the_plotted_commands(self):
        for side, preset, baseline, protected in (("measurement", "readout", "raw", "guard"),
                                                 ("execution", "delivery", "bare", "full"),
                                                 ("execution", "mixed", "bare", "full")):
            with self.subTest(preset=preset):
                engine = Comparison(Config.parse(dict(side=side, scenario=preset, seed=1301)), continuous=True)
                peak_gap = [0., 0.]
                second_iae = {baseline: 0., protected: 0.}
                starts, stops, responses = [], [], []
                last = None
                for k in range(286):
                    row, events = engine.advance("begin" if k in (10, 155) else None)
                    groups = row["groups"]
                    if last:
                        self.assertEqual(groups[protected]["previous_Qc"], last["Qc"])
                    last = groups[protected]
                    if row["phase"] == "active":
                        idx = row["segment"] - 1
                        gap = abs(groups[protected]["Qc_cmd"] - groups[baseline]["Qc_cmd"])
                        peak_gap[idx] = max(peak_gap[idx], gap)
                    if 155 <= k < 285:
                        for arm in second_iae:
                            sample = groups[arm]
                            second_iae[arm] += abs(sample["T"] - sample["T_ref"]) * .1
                    for event in events:
                        if event["event"] == "segment_start":
                            starts.append(k)
                        elif event["event"] == "segment_end":
                            stops.append(k)
                        elif event["event"] == "compensation_response":
                            responses.append(k)
                self.assertEqual(starts, [10, 155])
                self.assertEqual(stops, [60, 205])
                self.assertGreater(min(peak_gap), 0.1)
                self.assertLess(second_iae[protected], second_iae[baseline])
                if side == "execution":
                    self.assertEqual(responses, [11, 156])

    def test_live_clocks_outlast_the_scoring_window(self):
        with TemporaryDirectory() as tmp:
            service = FdiService(Path(tmp))
            try:
                for side in ("measurement", "execution"):
                    with self.subTest(side=side):
                        session = service.create({"side": side, "speed": 1200})
                        started = wait_for(session, lambda snap: snap["seq"] >= 1)
                        self.assertEqual(started["status"], "running")
                        session.command("begin", uuid4().hex, started["run"])
                        later = wait_for(session, lambda snap: snap["latest"]["phase"] == "monitoring")
                        session.command("pause", uuid4().hex, started["run"])
                        report = session.export()
                        end = report["stop_seq"] + RECOVERY_STEPS
                        self.assertEqual(later["status"], "running")
                        rows = {r["seq"]: r for r in report["records"] if r["kind"] == "sample"}
                        self.assertTrue(rows[end]["evaluation_end"])
                        self.assertFalse(rows[end]["terminal"])
                        self.assertEqual(sum(r.get("event") == "horizon_reached" for r in report["records"]), 1)
                        for name, sample in rows[end]["groups"].items():
                            self.assertIsNotNone(sample["Qc"])
                            self.assertEqual(rows[end+1]["groups"][name]["previous_Qc"], sample["Qc"])
                            self.assertNotEqual(rows[end+1]["groups"][name]["T"], sample["T"])
                        finite = Comparison(Config.parse(report["config"]))
                        scores = {arm: Scores() for arm in finite.branches}
                        for seq in range(end+1):
                            row, _ = finite.advance("begin" if seq == report["start_seq"] else None)
                            for arm, score in scores.items():
                                score.record(row["groups"][arm], row, finite.stop)
                        self.assertEqual(report["metrics"], {arm: score.report() for arm, score in scores.items()})
                        session.command("begin", uuid4().hex, started["run"])
                        session.command("resume", uuid4().hex, started["run"])
                        wait_for(session, lambda snap: snap["segment"] == 2)
                        segments = session.export()["segments"]
                        self.assertEqual(segments[0]["score_status"], "complete")
                        self.assertEqual(segments[0]["metrics"], report["metrics"])
                        self.assertGreater(segments[1]["start_seq"], end)
            finally:
                service.close()

    def test_repeat_while_active_and_during_recovery(self):
        with TemporaryDirectory() as tmp:
            service = FdiService(Path(tmp))
            try:
                for side, preset, second, arm in (("measurement", "readout", "readout", "guard"),
                                                 ("execution", "delivery", "mixed", "full")):
                    with self.subTest(side=side):
                        session = service.create({"side": side, "speed": 240}, auto_start=False)
                        run = wait_for(session, lambda s: s["seq"] == 0)["run"]

                        def send(kind, scenario=None):
                            settings = {"scenario": scenario} if scenario else None
                            return session.command(kind, uuid4().hex, run, settings)

                        send("begin", preset)
                        send("start")
                        wait_for(session, lambda s: s["seq"] >= 4)
                        send("pause")
                        paused = session.snapshot()["seq"]
                        send("begin", second)
                        send("finish")  # A cancelled repeat leaves the current exposure intact.
                        self.assertEqual(session.snapshot()["latest"]["phase"], "active")
                        key = uuid4().hex
                        request = ("begin", key, run, {"scenario": second})
                        self.assertEqual(session.command(*request), session.command(*request))
                        self.assertEqual(session.snapshot()["seq"], paused)
                        send("resume")
                        wait_for(session, lambda s: s["segment"] == 2 and s["seq"] >= paused + 5)
                        send("finish")
                        wait_for(session, lambda s: s["latest"]["phase"] == "recovery")
                        send("pause")
                        in_recovery = session.snapshot()["seq"]
                        send("begin", preset)
                        send("resume")
                        wait_for(session, lambda s: s["segment"] == 3)
                        send("pause")
                        report = session.export()
                        first, middle, third = report["segments"]
                        self.assertEqual([s["scenario"] for s in report["segments"]], [preset, second, preset])
                        self.assertEqual((first["stop_seq"], middle["start_seq"]), (paused+1, paused+2))
                        self.assertEqual(third["start_seq"], in_recovery+1)
                        self.assertEqual(first["metrics"][arm]["recovery_intervals"], 1)
                        self.assertEqual(middle["metrics"][arm]["event_intervals"], middle["stop_seq"]-middle["start_seq"])
                        self.assertEqual(middle["score_status"], "interrupted")
                        starts = [r for r in report["records"] if r.get("event") == "segment_start"]
                        self.assertEqual([r["segment"] for r in starts], [1, 2, 3])
                        gap = session.window(run, paused+1, 1)
                        self.assertEqual((gap["latest"]["phase"], gap["bands"]), ("recovery", []))
                        self.assertEqual(view(gap, live=False)["diagrams"][arm]["SIGNALLABEL"], "已选 · 待生效")
                        self.assertEqual(len(session.snapshot()["bands"]), 3)
                        send("reset")
                        archived = session.window(run, 0, middle["start_seq"]+1)
                        self.assertEqual((archived["segment"], archived["scenario"]), (2, second))
                        self.assertEqual(len(archived["bands"]), 2)
                        self.assertEqual(archived["metrics"], middle["metrics"])
                        self.assertEqual(view(archived, live=False)["diagrams"][arm]["SIGNALLABEL"],
                                         ("测量端" if side == "measurement" else "执行端")+" FDI 生效")
                        self.assertEqual(session.export(run)["segments"][:2], [first, middle])
                        self.assertEqual(session.snapshot()["segment"], 0)
            finally:
                service.close()

    def test_cancel_pause_early_end_reset_and_history(self):
        with TemporaryDirectory() as tmp:
            service = FdiService(Path(tmp))
            try:
                s = service.create({"speed":120}, auto_start=False)
                snap = wait_for(s, lambda x:x["seq"]==0)
                old = snap["run"]
                def send(kind, settings=None):
                    return s.command(kind, uuid4().hex, s.run, settings)
                send("begin")
                send("finish")
                self.assertEqual(view(s.snapshot())["diagrams"]["guard"]["MEASUREMARK"], "0")
                send("begin")
                send("resume")
                wait_for(s, lambda x:x["seq"]>=2)
                send("pause")
                paused = s.snapshot()
                at = paused["seq"]
                self.assertEqual(view(paused)["diagrams"]["guard"]["SIGNALLABEL"], "测量端 FDI 生效")
                send("finish")
                self.assertIsNone(s.snapshot()["stop_seq"])
                time.sleep(.065)
                self.assertEqual(s.snapshot()["seq"], at)
                send("resume")
                reached = wait_for(s, lambda x:x["stop_seq"] is not None)
                self.assertEqual(reached["stop_seq"], at+1)
                send("end")
                self.assertEqual(send("reset", {"scenario":"healthy"})["status"], "running")
                archived = s.window(old, 1, 1)
                self.assertEqual(view(archived,live=False)["diagrams"]["raw"]["SIGNALLABEL"], "测量端 FDI 生效")
                self.assertEqual(s.export(old)["stop_seq"], at+1)
                self.assertNotEqual(s.run, old)
                self.assertEqual(s.snapshot(run=old)["resync"], True)
                self.assertEqual(view(wait_for(s,lambda x:x["seq"]>=1))["diagrams"]["guard"]["MEASUREMARK"],"0")
            finally:
                service.close()
