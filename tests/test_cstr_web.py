"""HTTP ownership and the incremental CSTR session contract."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from platform_core.risk_control.service import ControlService
from platform_core.task_manager import TaskManager
from web_backend.app import app


class CstrWebTests(unittest.TestCase):
    def test_cstr_job_records_recover_from_private_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TaskManager(root/"common")
            service = ControlService(manager, root/"private")
            task = manager.submit("CSTR", lambda log: {"ok": True}, record_root=service.record_root)
            manager._executor.shutdown(wait=True)
            manager.shutdown()
            self.assertTrue((service.record_root/task.id/"task.json").is_file())
            self.assertFalse((root/"common"/task.id).exists())
            recovered = TaskManager(root/"common")
            ControlService(recovered, root/"private")
            try:
                self.assertEqual(recovered.get(task.id)["result"], {"ok": True})
            finally:
                recovered.shutdown()

    def test_owner_commands_incremental_reads_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TaskManager(root/"tasks")
            service = ControlService(manager, root/"cstr")
            with patch("web_backend.app.cstr_service", service):
                client = TestClient(app)
                try:
                    config = client.get("/api/cstr/config").json()
                    self.assertEqual(config["navigation"][0], "综合仿真展示")
                    self.assertEqual(len(config["stages"]), 2)
                    created = client.post("/api/cstr/sessions", json={}).json()
                    url = "/api/cstr/sessions/"+created["id"]
                    owner = {"X-Control-Owner": created["owner"]}
                    initial = client.get(url).json()
                    self.assertEqual([initial["latest"][k] for k in ("C", "T", "Tc")], config["nominal"])
                    history = client.get(url+"/history", params={"run": initial["run"], "first": 0, "count": 121})
                    self.assertEqual(history.json()["samples"][0]["seq"], 0)
                    self.assertEqual(client.get(url+"/presence").status_code, 403)
                    self.assertIn("trajectory,step,C,T,Tc,Qc,Ci,Ti,Tci", client.get("/api/cstr/training/example.csv").text)
                    self.assertEqual(client.post("/api/cstr/training/csv", content=b"").status_code, 400)
                    automatic = client.post("/api/cstr/sessions", json={"auto_start": True}).json()
                    auto_url = "/api/cstr/sessions/"+automatic["id"]
                    auto_snap = client.get(auto_url).json()
                    self.assertEqual(auto_snap["status"], "running")
                    self.assertTrue(initial["online_update"]["enabled"])
                    policy = auto_snap["online_update"]
                    self.assertTrue(policy["enabled"])
                    self.assertEqual((policy["device"], policy["batch_intervals"]), ("auto", 24))
                    self.assertIn("status", policy)
                    auto_owner = {"X-Control-Owner": automatic["owner"]}
                    change = {"enabled": False, "device": "cpu"}
                    endpoint = auto_url+"/updates/policy"
                    self.assertEqual(client.post(endpoint, json=change).status_code, 403)
                    for invalid in ({"enabled": "false"}, {"device": "missing"}):
                        self.assertEqual(client.post(endpoint, json={**change, **invalid}, headers=auto_owner).status_code, 422)
                    disabled = client.post(endpoint, json=change, headers=auto_owner)
                    self.assertEqual(disabled.status_code, 200)
                    self.assertFalse(disabled.json()["enabled"])
                    first = auto_snap["samples"][0]
                    self.assertEqual([first[k] for k in ("C", "T", "Tc")], config["demo_initial"])
                    self.assertEqual(first["T"], 432.)
                    self.assertEqual(auto_snap["reference"], 433.)
                    reset = {"kind": "reset", "key": "demo-reset"}
                    reset_reply = client.post(auto_url+"/commands", json=reset,
                                              headers={"X-Control-Owner": automatic["owner"]})
                    self.assertEqual(reset_reply.status_code, 200)
                    restarted = client.get(auto_url).json()
                    self.assertNotEqual(restarted["run"], auto_snap["run"])
                    self.assertEqual(restarted["samples"][0]["T"], first["T"])
                    self.assertEqual(restarted["status"], "running")
                    self.assertFalse(restarted["online_update"]["enabled"])
                    self.assertEqual(restarted["online_update"]["device"], "cpu")
                    enabled = client.post(endpoint, json={**change, "enabled": True}, headers=auto_owner)
                    self.assertTrue(enabled.json()["enabled"])
                    client.delete(auto_url, headers={"X-Control-Owner": automatic["owner"]})
                    self.assertEqual(client.get(url).json()["latest"]["feedback"], "ideal")
                    self.assertEqual(client.get(url+"?after=0").json()["samples"], [])
                    self.assertEqual([p["id"] for p in config["disturbances"]["presets"]],
                                     ["feed_concentration", "feed_temperature", "coolant_temperature"])
                    pulse = dict(kind="disturbance", preset="coolant_temperature", key="cool")
                    self.assertEqual(client.post(url+"/commands", json=pulse).status_code, 403)
                    for change in ({"preset": "other"}, {"duration_cycles": 0}, {"duration_cycles": 61},
                                   {"duration_cycles": True}):
                        self.assertGreaterEqual(client.post(url+"/commands", json={**pulse, **change}, headers=owner).status_code, 400)
                    self.assertEqual(client.post(url+"/commands", json=pulse, headers=owner).status_code, 200)
                    waiting = client.get(url).json()
                    self.assertEqual(waiting["disturbance"]["cycles"], 20)
                    self.assertIn("等待", waiting["disturbance_display"]["status"])
                    clear = dict(kind="clear_disturbance", key="clear")
                    self.assertEqual(client.post(url+"/commands", json=clear, headers=owner).status_code, 200)
                    self.assertEqual(client.get(url).json()["disturbance"]["phase"], "cancelled")
                    cmd = {"kind": "reference", "value": 432, "key": "reference"}
                    self.assertEqual(client.post(url+"/commands", json=cmd).status_code, 403)
                    self.assertEqual(client.post(url+"/commands", json=cmd, headers=owner).status_code, 200)
                    self.assertGreaterEqual(client.post(
                        url+"/commands", json={**cmd, "key": "bad", "value": 500}, headers=owner).status_code, 400)
                    self.assertGreaterEqual(client.post(url+"/updates", json={},
                                            headers=owner).status_code, 400)
                    self.assertEqual(client.get(url+"/export").json()["run"], created["run"])
                    self.assertEqual(client.delete(url, headers=owner).status_code, 200)
                    self.assertEqual(client.get(url).status_code, 404)
                finally:
                    client.close()
                    service.close()
                    manager.shutdown()
