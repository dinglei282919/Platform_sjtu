"""Scheduling and cancellation of session-owned online DNN updates."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np

from platform_core.risk_control import cstr
from platform_core.risk_control.service import ControlService
from platform_core.risk_control.session import Session, UPDATE_BATCH
from platform_core.task_manager import TaskManager


class OnlineSchedulerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.tasks = TaskManager(root/"tasks")
        self.addCleanup(self.tasks.shutdown)
        self.service = ControlService(self.tasks, root/"cstr")
        self.addCleanup(self.service.close)

        # These tests supply completed transitions explicitly. Only the sampler's
        # clock is parked; the session scheduler, job queue and journal are real.
        def park_clock(session):
            with session.lock:
                session.lock.wait_for(lambda: session.closed)

        sampler = patch.object(Session, "_loop", park_clock)
        sampler.start()
        self.addCleanup(sampler.stop)
        self.s = self.service.create(auto_start=True)
        self.service.configure_updates(self.s, True, "cpu")

    def advance(self, count):
        with self.s.lock:
            for _ in range(count):
                x = self.s.state
                y = cstr.step(x, self.s.previous)
                self.s.history.append((np.r_[x, self.s.previous, cstr.INLET], y))
                self.s.seq += 1
                self.s.state = y
            self.s.lock.notify_all()

    def wait(self, predicate):
        end = time.monotonic()+5
        while time.monotonic() < end:
            state = self.s.snapshot()
            if predicate(state):
                return state
            time.sleep(.01)
        self.fail(str(self.s.snapshot()["online_update"]))

    @staticmethod
    def requests(snap):
        return [e for e in snap["events"] if e["event"] == "update_requested"]

    def test_fresh_batches_single_flight_and_rejection_without_polling_to_schedule(self):
        entered, release = Event(), Event()

        def fit(*args, initial, **kwargs):
            entered.set()
            if not release.wait(4):
                raise TimeoutError("test gate")
            return copy.deepcopy(initial), []

        with patch("platform_core.risk_control.service.training.fit", side_effect=fit), patch(
            "platform_core.risk_control.service.validation.check", return_value={"passed": False, "reasons": ["gate"]}
        ):
            try:
                self.advance(UPDATE_BATCH-1)
                self.assertFalse(entered.wait(.07))
                self.advance(1)
                self.assertTrue(entered.wait(4))
                first = self.s.snapshot()
                self.assertEqual(first["online_update"]["phase"], "training")
                self.assertEqual(self.service.update_model(self.s, "cpu"), first["update"])
                self.advance(UPDATE_BATCH+7)
                self.assertEqual(len(self.requests(self.s.snapshot())), 1)
            finally:
                release.set()
            done = self.wait(lambda s: len(self.requests(s)) == 2 and s["update"] is None)
            jobs = self.requests(done)
            self.assertEqual([j["cutoff"] for j in jobs], [UPDATE_BATCH, 2*UPDATE_BATCH+7])
            self.assertTrue(all(j["automatic"] for j in jobs))
            self.assertEqual(done["online_update"]["last"]["state"], "rejected")
            self.assertEqual(done["model"], first["model"])
            self.advance(UPDATE_BATCH-1)
            self.assertEqual(self.s.snapshot()["online_update"]["new_intervals"], UPDATE_BATCH-1)
            self.advance(1)
            self.wait(lambda s: len(self.requests(s)) == 3 and s["update"] is None)

    def test_disabling_discards_ready_automatic_candidate_and_survives_reset(self):
        self.s.command("pause", "pause")
        self.advance(UPDATE_BATCH)
        info, _ = self.s.prepare_update(automatic=True)
        self.assertTrue(self.s.finish_update(info, self.s.model, {"passed": True}))
        self.assertEqual(self.s.snapshot()["online_update"]["phase"], "awaiting_cycle")
        self.service.configure_updates(self.s, False, "cpu")
        self.assertTrue(info["cancel"].is_set())
        self.assertIsNone(self.s.pending)
        self.assertFalse(self.s.finish_update(info, self.s.model, {"passed": True}))
        self.s.command("reset", "reset")
        state = self.s.snapshot()["online_update"]
        self.assertFalse(state["enabled"])
        self.assertEqual((state["device"], state["new_intervals"]), ("cpu", 0))
        self.advance(UPDATE_BATCH)
        self.assertEqual(self.requests(self.s.snapshot()), [])
        # A manual one-off remains eligible with automation switched off.
        manual, _ = self.s.prepare_update()
        self.assertFalse(manual["automatic"])
        self.assertTrue(self.s.finish_update(manual, self.s.model, {"passed": True}))
        self.service.configure_updates(self.s, False, "auto")
        self.assertIs(self.s.pending[1], manual)

    def test_registration_failure_retries_with_new_data_and_keeps_control_running(self):
        with patch.object(self.service, "_job", side_effect=OSError("registration failed")) as submit:
            self.advance(UPDATE_BATCH)
            state = self.wait(lambda s: (s["online_update"]["last"] or {}).get("state") == "failed")
            self.assertEqual(state["status"], "running")
            self.assertIsNone(state["update"])
            self.assertIn("registration failed", state["online_update"]["last"]["detail"])
            self.advance(UPDATE_BATCH-1)
            time.sleep(.06)
            self.assertEqual(submit.call_count, 1)
            self.advance(1)
            self.wait(lambda s: sum(e["event"] == "update_failed" for e in s["events"]) == 2)

    def test_reset_while_task_is_registering_never_activates_stale_weights(self):
        entered, release = Event(), Event()
        original = self.service._job

        def slow_submit(*args, **kwargs):
            entered.set()
            if not release.wait(4):
                raise TimeoutError("test gate")
            return original(*args, **kwargs)

        with patch.object(self.service, "_job", side_effect=slow_submit), patch(
            "platform_core.risk_control.service.training.fit"
        ) as fit:
            try:
                self.advance(UPDATE_BATCH)
                self.assertTrue(entered.wait(4))
                self.s.command("reset", "new-run")
                self.assertEqual(self.s.snapshot()["seq"], 0)
            finally:
                release.set()
            with self.s.update_submit_lock:
                pass
            self.service.close()
            self.tasks._executor.shutdown(wait=True)
            fit.assert_not_called()
            self.assertFalse(self.s.update_worker.is_alive())
            self.assertEqual(self.requests(self.s.snapshot()), [])

