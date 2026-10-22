import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from web_backend import app as server
from web_backend.task_manager import TaskManager

class ReactorApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = TaskManager(Path(self.temp.name) / "tasks")
        self.patch = patch.object(server, "task_manager", self.manager)
        self.patch.start()
        self.original_new_dir = server.reactor.new_run_dir
        self.dirs = patch.object(server.reactor, "new_run_dir", self.new_dir)
        self.dirs.start()
        self.client = TestClient(server.app)

    def new_dir(self):
        from uuid import uuid4
        path = Path(self.temp.name) / uuid4().hex
        path.mkdir()
        return path

    def tearDown(self):
        self.manager.shutdown()
        self.dirs.stop()
        self.patch.stop()
        self.temp.cleanup()

    def wait(self, task_id):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            task = self.client.get("/api/tasks/" + task_id).json()
            if task["status"] not in {"running", "queued"}:
                return task
            time.sleep(.02)
        self.fail("task did not finish")

    def test_defaults_and_validation(self):
        c = self.client.get("/api/reactor/defaults").json()
        self.assertEqual(c["duration"], 3)
        self.assertEqual(self.client.post("/api/reactor/tasks", json={**c, "prescribed_time": 3.1}).status_code, 422)
        self.assertEqual(self.client.post("/api/reactor/tasks", json={**c, "ell": .01}).status_code, 422)
        self.assertEqual(self.client.post("/api/reactor/tasks", json={"unknown": 1}).status_code, 422)
        self.assertEqual(self.client.get("/api/reactor/diagram/reactor_dual_feed.png").headers["content-type"], "image/png")
        self.assertEqual(self.client.get("/api/reactor/diagram/not-allowed.txt").status_code, 404)

    def test_task_trajectory_and_downloads(self):
        response = self.client.post("/api/reactor/tasks", json={"duration": .3, "prescribed_time": .2})
        self.assertEqual(response.status_code, 200, response.text)
        task_id = response.json()["task_id"]
        task = self.wait(task_id)
        self.assertEqual(task["status"], "succeeded", task)
        data = self.client.get(f"/api/reactor/tasks/{task_id}/trajectory").json()
        self.assertEqual(data["t"][-1], .3)
        for name in server.reactor.ARTIFACTS:
            response = self.client.get(f"/api/reactor/tasks/{task_id}/files/{name}")
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.content), 100)
        self.assertEqual(self.client.get(f"/api/reactor/tasks/{task_id}/files/progress.json").status_code, 404)
        self.assertEqual(task["progress"]["percent"], 100)

    def test_cancel_and_failed_task_no_results(self):
        task_id = self.client.post("/api/reactor/tasks", json={}).json()["task_id"]
        self.assertEqual(self.client.post(f"/api/tasks/{task_id}/cancel").status_code, 200)
        self.assertEqual(self.wait(task_id)["status"], "cancelled")
        self.assertEqual(self.client.get(f"/api/reactor/tasks/{task_id}/files/results.png").status_code, 409)

    def test_cross_module_artifacts_rejected(self):
        directory = self.new_dir()
        task = self.manager.submit("different", lambda log: {"module": "replay"}, artifact_dir=directory)
        self.wait(task.id)
        self.assertEqual(self.client.get(f"/api/reactor/tasks/{task.id}/trajectory").status_code, 404)

    def test_failure_exposed(self):
        with patch.object(server.reactor, "simulate", side_effect=RuntimeError("solver failure")):
            task_id = self.client.post("/api/reactor/tasks", json={}).json()["task_id"]
            task = self.wait(task_id)
        self.assertEqual(task["status"], "failed")
        self.assertIn("solver failure", task["error"])

    def test_default_configuration_matches_shared_core(self):
        from dataclasses import asdict
        from platform_core.reactor.core import Config
        expected = asdict(Config())
        exposed = self.client.get("/api/reactor/defaults").json()
        self.assertEqual(exposed, {key: expected[key] for key in exposed})
        self.assertEqual(set(exposed), {"duration", "sample_dt", "prescribed_time", "ell",
                                       "x1", "x2", "observer1", "observer2"})

    def test_output_directory_uses_platform_convention(self):
        from platform_core.control_paths import module_output_dir
        with patch.object(server.reactor, "module_output_dir", return_value=Path(self.temp.name)/"reactor") as output:
            directory = self.original_new_dir()
        output.assert_called_once_with("reactor")
        self.assertEqual(directory.parent, Path(self.temp.name)/"reactor"/"runs")
        self.assertTrue(directory.is_dir())
        self.assertEqual(module_output_dir("reactor").name, "reactor")
