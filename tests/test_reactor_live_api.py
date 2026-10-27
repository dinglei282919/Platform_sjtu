import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from web_backend import app as server
from web_backend.services.reactor import ReactorLiveRegistry


class ReactorLiveApiTests(unittest.TestCase):
    def setUp(self):
        self.registry = ReactorLiveRegistry(lease_seconds=.4, retention_seconds=10)
        self.override = patch.object(server.reactor, "live_sessions", self.registry)
        self.override.start()
        self.client = TestClient(server.app)
        response = self.client.post("/api/reactor/live", json={})
        self.assertEqual(response.status_code, 200, response.text)
        self.id = response.json()["id"]
        self.url = "/api/reactor/live/" + self.id

    def tearDown(self):
        self.client.close()
        self.registry.close()
        self.override.stop()

    def wait(self, predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            snapshot = self.client.get(self.url).json()
            if predicate(snapshot):
                return snapshot
            time.sleep(.02)
        self.fail("live snapshot did not reach expected state")

    def command(self, kind, value=None):
        result = self.client.post(self.url + "/command", json={"kind": kind, "value": value})
        self.assertEqual(result.status_code, 200, result.text)
        sequence = result.json()["queued_sequence"]
        return self.wait(lambda value: value["applied_sequence"] >= sequence)

    def test_normal_state_events_and_independent_session(self):
        normal = self.wait(lambda value: value["t"] >= .1)
        self.assertIn('xs',normal['data'])
        changed = self.command("r2", .4)
        self.assertEqual(changed['controls']['r2'],.4)
        self.assertTrue(any(e['kind']=='r2' for e in changed['events']))
        other=self.client.post('/api/reactor/live').json()
        self.assertNotEqual(other['id'],self.id)
        self.assertEqual(other['controls']['r2'],.2)
        self.assertEqual(other['data']['xs'][0],1.)
        for key in ('xs','xd','observer_s','observer_d','received_s','received_d','r1','r2','k'):
            self.assertEqual(len(changed['data'][key]),len(changed['data']['t']))

    def test_pause_resume_and_explicit_reset(self):
        paused = self.command("pause")
        time.sleep(.12)
        self.assertEqual(self.client.get(self.url).json()["t"], paused["t"])
        self.command("resume")
        self.wait(lambda value: value["t"] > paused["t"])
        self.command("r1", .4)
        reset = self.command("reset")
        self.assertEqual(reset["controls"]["r1"], .4)
        self.assertEqual(reset["controls"]["mode"], 0)
        self.assertEqual(reset["data"]["xs"][0], 1.)

    def test_bad_controls_leave_the_running_session_intact(self):
        for command in ({"kind": "r2", "value": 1.}, {"kind": "mode", "value": 3},
                        {"kind": "enabled", "value": "false"}, {"kind": "impulse", "value": True},
                        {"kind": "reset", "value": 1}, {"kind": "unknown"}):
            self.assertEqual(self.client.post(self.url + "/command", json=command).status_code, 422)
        current = self.client.get(self.url).json()
        self.assertEqual(current["controls"]["r2"], .2)
        self.assertNotEqual(current["status"], "error")

    def test_attack_settings_and_switch_with_validation(self):
        changed=self.command('attack',dict(scale=1.7,frequency=.6,phase=.4))
        self.assertEqual(changed['controls']['attack_scale'],1.7)
        self.assertTrue(any(e['kind']=='attack' for e in changed['events']))
        off=self.command('attack_enabled',False)
        self.assertEqual(off['data']['k'][-1],1.)
        self.assertFalse(off['controls']['attack_enabled'])
        self.command('attack_enabled',True)
        for value in (dict(scale=2.,frequency=.5,phase=0.),dict(scale=1.,frequency=0.,phase=0.),
                      dict(scale=1.,frequency=.5,phase=4.),dict(scale=True,frequency=.5,phase=0.),dict(scale=1.)):
            response=self.client.post(self.url+'/command',json=dict(kind='attack',value=value))
            self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.client.post(self.url+'/command',json=dict(kind='attack_enabled',value=1)).status_code,422)

    def test_disconnected_client_pauses_and_delete_releases_runtime(self):
        runtime = self.registry.get(self.id)
        deadline = time.monotonic() + 3
        while runtime.snapshot()["status"] != "paused" and time.monotonic() < deadline:
            time.sleep(.03)
        self.assertEqual(runtime.snapshot()["status"], "paused")
        self.assertEqual(self.client.get(self.url).json()["status"], "paused")
        self.assertEqual(self.client.delete(self.url).status_code, 200)
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertFalse(runtime._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
