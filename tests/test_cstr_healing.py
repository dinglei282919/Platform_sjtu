"""Module 2 sample boundaries, four independent plants and live ownership."""
import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

from platform_core.risk_control.cstr import NOMINAL
from platform_core.risk_control.healing import Comparison, Config, GROUPS, TOLERANCE, fit_maps, options
from platform_core.risk_control.healing_display import effect_view
from platform_core.risk_control.healing_session import HealingService, Metrics
from web_backend.app import app, cstr_service


class ComparisonTests(unittest.TestCase):
    def test_fitted_normal_maps_and_first_applied_intervals(self):
        diagnostic = fit_maps(42)[-1]
        self.assertEqual((diagnostic["samples"], diagnostic["input_rank"], diagnostic["output_rank"]), (5000, 3, 3))
        for scenario, event_step, batch_step, channel in (("catalyst",2110,2200,"C"), ("htc",3379,3500,"Tc")):
            for mode, expected in (("event",event_step),("batch",batch_step)):
                engine = Comparison(Config.parse(dict(scenario=scenario,mode=mode)))
                first = {}
                last = {}
                for _ in range(expected+55):
                    row, state = engine.prepare()
                    if row["seq"] <= 2000:
                        self.assertEqual(row["groups"]["normal"],row["groups"]["scsc"])
                    self.assertAlmostEqual(row["groups"]["normal"]["T"],NOMINAL[1],places=9)
                    for action in row["compensation"]:
                        key = action["group"]
                        first.setdefault(key, action)
                        self.assertEqual(action["first_seq"],row["seq"])
                        np.testing.assert_array_equal(action["inlets"],[row["groups"][key][s] for s in ("Ci","Ti","Tci")])
                        if key in last and mode=="event":
                            self.assertGreaterEqual(action["seq"]-last[key],round(getattr(engine.config,key+"_interval")*10))
                        if mode=="batch":
                            self.assertEqual(action["seq"]%100,0)
                        last[key]=action["seq"]
                    engine.commit(state)
                for g in ("scsc","ccsc"):
                    self.assertEqual((first[g]["seq"],first[g]["channels"]), (expected,[channel]))
                    self.assertEqual(first[g]["first_seq"],expected+1)

    def test_trigger_width_gap_and_signed_gains(self):
        for invalid in ({"widths":[0.,1.,1.]},{"widths":[1.,float("nan"),1.]},{"widths":[1,2]},
                        {"scsc_interval":.15},{"ccsc_interval":0},{"scsc_gain":float("inf")},{"seed":True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Config.parse(invalid)
        conf=Config.parse(dict(scenario="htc",widths=[.0015,.5,.7],ccsc_gain=.01))
        self.assertEqual(conf.widths,(.0015,.5,.7))
        self.assertEqual((conf.ccsc_interval,conf.ccsc_gain),(5.,.01))
        self.assertEqual(set(options()["modes"]), {"event", "batch"})
        with self.assertRaises(ValueError):
            Config.parse(dict(scenario="htc", mode="batch_retuned"))
        self.assertEqual(Config.parse(dict(scenario="htc", mode="batch")).ccsc_gain, -.3)
        self.assertEqual(Comparison(Config.parse({"ccsc_interval":1500.})).gaps[3], 15000)

    def test_channel_limits_match_the_accepted_endpoints(self):
        limits = options()["trigger_limits"]
        for field, kind, label in (("references", "reference", "参考点"), ("widths", "half_width", "半宽")):
            for i, signal in enumerate(("C", "T", "Tc")):
                lo, hi = limits[signal][kind]
                values = list(getattr(Config(), field))
                self.assertLess(lo, values[i])
                self.assertLess(values[i], hi)
                for endpoint in (lo, hi):
                    values[i] = endpoint
                    with self.subTest(field=field, channel=signal, value=endpoint):
                        self.assertEqual(getattr(Config.parse({field: values}), field)[i], endpoint)
                for invalid in (math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)):
                    values[i] = invalid
                    with self.subTest(field=field, channel=signal, value=invalid):
                        with self.assertRaisesRegex(ValueError, f"{signal} {label}"):
                            Config.parse({field: values})

    def test_editable_references_drive_detection_and_both_corrections(self):
        for values in ([0, 431, 417], [.1, float("inf"), 417], [.1, 431], [False, 431, 417]):
            with self.assertRaises(ValueError):
                Config.parse({"references": values})
        reference = np.array([.1008, 431.1, 416.95])
        for mode, boundary in (("event", 1), ("batch", 100)):
            engine = Comparison(Config.parse(dict(mode=mode, references=reference.tolist(), widths=[.0001, .1, .1])), fault_seq=None)
            self.assertEqual(engine.initial()["groups"]["scsc"]["outside"], ["C", "T", "Tc"])
            for _ in range(boundary):
                row, next_state = engine.prepare()
                self.assertFalse(row["compensation"])
                engine.commit(next_state)
            measured = engine.state[2] if mode == "event" else engine.batch[:, 2].mean(axis=0)
            error = measured - reference
            pearson, cca, dx, dy, _ = engine.maps
            expected_scsc = -.3 * ((error / reference * (1., 1., .05)) @ pearson.T) * engine.inlets[2]
            expected_ccsc = -.3 * ((error / dy) @ cca) * dx
            row, _ = engine.prepare()
            self.assertEqual([a["group"] for a in row["compensation"]], ["scsc", "ccsc"])
            for action, expected in zip(row["compensation"], (expected_scsc, expected_ccsc)):
                np.testing.assert_allclose(action["requested"], expected, atol=1e-10)
                np.testing.assert_allclose(action["error"], error, atol=1e-10)
            self.assertAlmostEqual(row["groups"]["normal"]["T"], NOMINAL[1], places=9)

    def test_joint_percentage_uses_healthy_concentration_and_reactor_temperature(self):
        engine = Comparison(Config.parse({"references": [.099, 430.7, 416.5]}))
        row = dict(engine.initial(), cycle_ms=1., wall_ms=10.)
        m = Metrics()
        m.record(row, 10., None)
        self.assertIsNone(m.summary()["groups"]["normal"]["joint_percent"])
        row["groups"]["normal"].update(C=NOMINAL[0]+TOLERANCE[0], T=NOMINAL[1]-TOLERANCE[1], Tc=400.)
        m.record(dict(row, seq=1, t=.1), 10., 0)
        row["groups"]["normal"]["C"] += .00001
        m.record(dict(row, seq=2, t=.2), 10., 0)
        self.assertEqual(m.summary()["post_fault_samples"], 2)
        self.assertEqual(m.summary()["groups"]["normal"]["joint_percent"], 50.)

class HealingSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service=HealingService(Path(self.tmp.name))
        self.addCleanup(self.service.close)

    def test_commit_pause_reset_deduplicate_and_archive(self):
        s=self.service.create(auto_start=False)
        with s.lock:
            for k in range(2212):
                s._advance(wall_ms=10.)
                if (k+1) % 512 == 0:
                    self.assertTrue(s.log.barrier().wait(5))
        snap=s.snapshot()
        self.assertEqual(len(snap["samples"]),640)
        self.assertEqual(len(s.history(s.run,0,640)),640)
        before=s.engine.state.copy()
        integrals=s.engine.integral.copy()
        batch=s.engine.batch.copy()
        held=s.engine.inlets.copy()
        last=s.engine.last_update.copy()
        previous=s.run
        s.command("pause","p",previous)
        self.assertEqual(s.snapshot(snap["seq"])["samples"],[])
        np.testing.assert_array_equal(before,s.engine.state)
        np.testing.assert_array_equal(integrals,s.engine.integral)
        np.testing.assert_array_equal(batch,s.engine.batch)
        np.testing.assert_array_equal(held,s.engine.inlets)
        self.assertEqual(last,s.engine.last_update)
        with self.assertRaises(ValueError):
            s.command("reset","bad",previous,{"widths":[1,0,1]})
        self.assertEqual(s.run,previous)
        exported=s.export()
        self.assertTrue(any(r.get("event")=="command_rejected" and r["key"]=="bad" for r in exported["records"]))
        self.assertEqual(len([r for r in exported["records"] if r["kind"]=="sample"]),2213)
        self.assertEqual(exported["metrics"]["timing"]["compensation"]["count"],len(set(e["seq"] for e in snap["recent"])))
        receipt=s.command("reset","repeat",previous,{"scenario":"htc","speed":1})
        self.assertEqual(s.command("reset","repeat",previous),receipt)
        s.command("pause","p2",s.run)
        self.assertNotEqual(s.run,previous)
        self.assertEqual(s.engine.seq,0)
        self.assertEqual(s.engine.last_update,[None]*4)
        self.assertTrue(s.snapshot(2212,run=previous)["resync"])
        with self.assertRaises(ValueError):
            s.history(previous,0,2)

    def test_end_freezes_the_journal_and_releases_its_writer(self):
        s = self.service.create(auto_start=False)
        old = s.run
        s.command("end", "end", old)
        self.assertFalse(s.log.thread.is_alive())
        self.assertEqual(s.export()["records"][-1]["event"], "command_applied")
        self.assertEqual(s.history(old, 0, 1)[0]["seq"], 0)
        s.command("reset", "reset", old, {"speed": 1})
        self.assertTrue(s.log.thread.is_alive())
        self.assertNotEqual(old, s.run)

    def test_failed_journal_enqueue_preserves_all_four_branches(self):
        s = self.service.create(auto_start=False)
        state = s.engine.state.copy()
        with s.lock, patch.object(s.log, "append", side_effect=RuntimeError("记录缓冲已满")):
            with self.assertRaises(RuntimeError):
                s._advance(wall_ms=10.)
        self.assertEqual(s.engine.seq, 0)
        np.testing.assert_array_equal(s.engine.state, state)
        self.assertEqual(s.snapshot()["latest"]["seq"], 0)

    def test_manual_onset_effect_trace_and_actual_compensation(self):
        s = self.service.create({"ccsc_gain": 0.}, auto_start=False)
        def advance(count):
            with s.lock:
                for _ in range(count):
                    s._advance(wall_ms=10.)
                    if s.engine.seq % 512 == 0:
                        self.assertTrue(s.log.barrier().wait(5))

        advance(2053)
        healthy = s.snapshot()
        self.assertIsNone(healthy["fault_min"])
        self.assertIsNone(healthy["metrics"]["groups"]["fault"]["rmse"])
        self.assertLessEqual(len(healthy["effect"]["points"]), 12)
        self.assertEqual(healthy["latest"]["groups"]["normal"], healthy["latest"]["groups"]["scsc"])
        s.command("pause", "pause", s.run)
        started = s.command("degrade", "inject", s.run)
        self.assertEqual(started, s.command("degrade", "inject", s.run))
        self.assertEqual(started["status"], "paused")
        with self.assertRaises(ValueError):
            s.command("degrade", "another-click", s.run)
        advance(1000)
        snap = s.snapshot()
        self.assertAlmostEqual(snap["fault_min"], 205.3)
        self.assertEqual(s.metrics.count, 1000)
        self.assertEqual(snap["latest"]["groups"]["normal"]["a"], 1.)
        factors = [snap["latest"]["groups"][g]["a"] for g in ("fault", "scsc", "ccsc")]
        self.assertTrue(factors[0] < 1 and factors.count(factors[0]) == 3)
        first = snap["effect"]["first"]["scsc"]
        self.assertAlmostEqual(first["at"] - snap["fault_min"], 11.)
        self.assertIsNone(snap["effect"]["first"]["ccsc"])
        self.assertGreater(snap["metrics"]["groups"]["ccsc"]["updates"], 0)
        points = snap["effect"]["points"]
        times = [p[0] for p in points]
        self.assertEqual(times, sorted(set(times)))
        for at in (snap["fault_min"], first["at"], first["first_sample_min"], snap["latest"]["t"]):
            self.assertIn(at, times)
        reference = snap["effect"]["reference"]
        for column, group in enumerate(("fault", "scsc", "ccsc"), 1):
            self.assertAlmostEqual(points[-1][column], snap["latest"]["groups"][group]["C"]-reference)
        self.assertLess(abs(points[-1][2]), abs(points[-1][1]))
        self.assertEqual(points[-1][1], points[-1][3])
        self.assertIn("减小", effect_view(snap)["methods"]["scsc"]["readout"])
        report = s.export()
        self.assertEqual(report["fault_min"], started["t"])
        self.assertAlmostEqual(report["end_min"], 1205.3)
        self.assertEqual(sum(e.get("command") == "degrade" and e.get("event") == "command_applied"
                             for e in report["records"]), 1)
        s.command("reset", "restart", s.run, {"speed": 1.})
        self.assertIsNone(s.snapshot()["fault_min"])
        self.assertIsNone(s.snapshot()["effect"]["first"]["scsc"])

    def test_http_owner_and_shared_signals(self):
        fit_maps(42)
        client = TestClient(app)
        self.addCleanup(client.close)
        with patch.object(cstr_service, "healing", self.service):
            base="/api/cstr/healing"
            contract = client.get(base+"/config").json()
            self.assertEqual(contract["fault_mode"], "manual")
            self.assertEqual(contract["trigger_limits"]["C"], {"reference": [.099,.101], "half_width": [.0001,.002]})
            self.assertEqual(contract["trigger_limits"]["T"], {"reference": [430.7,431.1], "half_width": [.1,3.]})
            self.assertEqual(contract["trigger_limits"]["Tc"], {"reference": [416.5,417.], "half_width": [.1,3.]})
            self.assertEqual(client.post(base+"/sessions",json={"settings":{"references":[.1,1.,417.]}}).status_code,400)
            created=client.post(base+"/sessions",json={"auto_start":False}).json()
            route=base+"/sessions/"+created["id"]
            headers={"X-Control-Owner":created["owner"]}
            command={"kind":"start","key":"start","run":created["run"]}
            invalid = client.post(route+"/commands",headers=headers,
                                  json={**command,"kind":"reset","key":"bad-range","settings":{"widths":[.0005,1.,3.01]}})
            self.assertEqual(invalid.status_code,400)
            self.assertIn("Tc 半宽",invalid.json()["detail"])
            self.assertEqual(client.get(route).json()["run"],created["run"])
            self.assertEqual(client.post(route+"/commands",json=command).status_code,403)
            self.assertEqual(client.get(route+"/presence").status_code,403)
            self.assertEqual(client.post(route+"/commands",json={**command,"kind":"degrade"}).status_code,403)
            self.assertEqual(client.post(route+"/commands",headers=headers,json=command).status_code,200)
            refs = [.1002, 431., 417.]
            accepted=client.post(route+"/commands",headers=headers,json={**command,"kind":"reset","key":"fast","settings":{"speed":1200, "references":refs}}).json()
            until=time.monotonic()+10
            while time.monotonic()<until:
                data=client.get(route).json()
                if data["seq"]>=2: break
                time.sleep(.03)
            self.assertGreaterEqual(data["seq"],2)
            self.assertIsNone(data["fault_min"])
            for group in GROUPS:
                self.assertEqual(data["view"]["diagrams"][group]["TEMP"],f"{data['latest']['groups'][group]['T']:.2f}")
            self.assertEqual(client.post(route+"/commands",json={"kind":"pause","key":"p","run":accepted["run"]},headers=headers).status_code,200)
            response = client.post(route+"/commands",json={"kind":"degrade","key":"manual","run":accepted["run"]},headers=headers)
            self.assertEqual(response.status_code, 200)
            report=client.get(route+"/export").json()
            self.assertEqual(report["trigger_reference"], refs)
            self.assertEqual(report["recovery_reference"], NOMINAL.tolist())
            self.assertEqual(report["output_reference"], NOMINAL.tolist())
            self.assertEqual(report["fault_min"], response.json()["t"])
            self.assertEqual(report["config"]["speed"],1200)
            self.assertEqual(report["mapping"]["samples"],5000)
            self.assertEqual(report["pi"]["flow_limits"],[10.,200.])
            self.assertEqual(report["recovery_tolerance_CT"],[.001,.5])
            self.assertEqual(client.delete(route,headers=headers).status_code,200)
            self.assertEqual(client.get(route).status_code,404)
