import time
import unittest
from platform_core.risk_control.live_runtime import LiveControlRuntime


class CounterSession:
    def __init__(self):
        self.t = 0.
        self.value = 0

    def advance(self, dt, cancelled=None):
        self.t += dt

    def apply_event(self, kind, value):
        if kind == "reset":
            self.t = 0.
        else:
            self.value = value

    def snapshot(self):
        return {"t": self.t, "controls": {"value": self.value}, "data": {"t": [self.t]}}


class LiveRuntimeTests(unittest.TestCase):
    def wait(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail("worker did not reach expected state")

    def test_clock_is_independent_of_polling_and_pause_preserves_state(self):
        runtime = LiveControlRuntime(CounterSession, interval=.02)
        try:
            self.wait(lambda: runtime.snapshot()["t"] >= .04)
            sequence = runtime.command("pause")
            self.wait(lambda: runtime.snapshot()["applied_sequence"] == sequence)
            before = runtime.snapshot()
            for _ in range(30):
                self.assertEqual(runtime.snapshot()["t"], before["t"])
            time.sleep(.06)
            self.assertEqual(runtime.snapshot()["t"], before["t"])
            sequence = runtime.command("setpoint", .2)
            self.wait(lambda: runtime.snapshot()["applied_sequence"] == sequence)
            self.assertEqual(runtime.snapshot()["controls"]["value"], .2)
            runtime.command("resume")
            self.wait(lambda: runtime.snapshot()["t"] > before["t"])
        finally:
            self.assertTrue(runtime.close())

    def test_lost_client_lease_pauses_and_readers_cannot_mutate_state(self):
        runtime = LiveControlRuntime(CounterSession, interval=.01, lease_seconds=.06)
        try:
            self.wait(lambda: runtime.snapshot()["status"] == "paused")
            before = runtime.snapshot()["t"]
            runtime.touch()
            snapshot = runtime.snapshot()
            snapshot["data"]["t"][0] = -999
            self.assertEqual(runtime.snapshot()["t"], before)
            self.assertNotEqual(runtime.snapshot()["data"]["t"][0], -999)
            self.assertEqual(runtime.snapshot()["status"], "paused")
        finally:
            self.assertTrue(runtime.close())

    def test_solver_failure_requires_reset_and_close_is_idempotent(self):
        class BrokenOnce(CounterSession):
            broken = True
            def advance(self, dt, cancelled=None):
                if self.broken:
                    self.broken = False
                    raise FloatingPointError("numerical failure")
                super().advance(dt)
        runtime = LiveControlRuntime(BrokenOnce, interval=.01)
        try:
            self.wait(lambda: runtime.snapshot()["status"] == "error")
            self.assertIn("numerical failure", runtime.snapshot()["message"])
            runtime.command("reset")
            self.wait(lambda: runtime.snapshot()["status"] == "running")
            self.wait(lambda: runtime.snapshot()["t"] > 0)
        finally:
            self.assertTrue(runtime.close())
            self.assertTrue(runtime.close())

    def test_event_failure_is_published_and_worker_can_reset(self):
        class FailedEvent(CounterSession):
            def apply_event(self, kind, value):
                if kind == "setpoint":
                    raise FloatingPointError("event overflow")
                super().apply_event(kind, value)
        runtime = LiveControlRuntime(FailedEvent, interval=.02)
        try:
            runtime.command("setpoint", .2)
            self.wait(lambda: runtime.snapshot()["status"] == "error")
            self.assertIn("event overflow", runtime.snapshot()["message"])
            self.assertTrue(runtime._thread.is_alive())
            runtime.command("reset")
            self.wait(lambda: runtime.snapshot()["status"] == "running")
        finally:
            self.assertTrue(runtime.close())


if __name__ == "__main__":
    unittest.main()
