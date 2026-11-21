"""Focused regression tests for the local B/S first-phase services."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import json
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from platform_core.scoring import default_values, evaluate
from web_backend.task_manager import TaskManager, TaskStateUnavailable
from web_backend.app import AnomalyTaskRequest, SDGNodeRequest, SILTaskRequest, app


class TaskManagerPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.managers = []

    def manager(self):
        manager = TaskManager(self.root / "tasks")
        self.managers.append(manager)
        return manager

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown()
            manager._executor.shutdown(wait=True)
        self.temporary.cleanup()

    def wait(self, manager, task_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            task = manager.get(task_id)
            if task["status"] in {"succeeded", "failed", "cancelled"}:
                return task
            time.sleep(.01)
        self.fail("Task did not finish")

    def test_construction_and_shutdown_preserve_existing_files(self):
        first = self.manager()
        self.assertFalse((self.root / "tasks").exists(), "Import/construction must not write to disk")
        original = self.root / "tasks" / "legacy" / "result.bin"
        original.parent.mkdir(parents=True)
        original.write_bytes(b"historical result")
        second = self.manager()
        first.shutdown()
        second.shutdown()
        self.assertEqual(original.read_bytes(), b"historical result")

    def test_completed_record_and_artifact_routes_survive_restart(self):
        manager = self.manager()
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        report = artifacts / "result.json"
        report.write_text(json.dumps({"answer": 42}), encoding="utf-8")
        created = manager.submit("CSTR 预测模型训练", lambda log: {"answer": 42}, artifact_dir=artifacts)
        finished = self.wait(manager, created.id)
        original = (created.work_dir / "task.json").read_bytes()
        manager.shutdown()
        recovered = self.manager()
        self.assertEqual(recovered.get(created.id)["result"], finished["result"])
        self.assertEqual(recovered.artifact_dir(created.id), artifacts)
        with patch("web_backend.app.task_manager", recovered), TestClient(app) as client:
            self.assertEqual(client.get(f"/api/tasks/{created.id}").json()["status"], "succeeded")
            response = client.get(f"/api/cstr/training/{created.id}/report")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, report.read_bytes())
        self.assertEqual((created.work_dir / "task.json").read_bytes(), original)

    def test_second_instance_does_not_invalidate_a_live_owner(self):
        ready, release = Event(), Event()
        def operation(log):
            ready.set()
            release.wait(10)
            return {"ok": True}
        owner = self.manager()
        created = owner.submit("live", operation)
        try:
            self.assertTrue(ready.wait(5))
            original = (created.work_dir / "task.json").read_bytes()
            reader = self.manager()
            self.assertEqual(reader.get(created.id)["status"], "running")
            with self.assertRaisesRegex(ValueError, "另一服务实例"):
                reader.cancel(created.id)
            reader.shutdown()
            self.assertEqual((created.work_dir / "task.json").read_bytes(), original)
        finally:
            release.set()
        self.assertEqual(self.wait(owner, created.id)["status"], "succeeded")

    def test_crashed_owner_is_reported_interrupted_without_deleting_artifacts(self):
        code = """
import sys
from pathlib import Path
from threading import Event
from web_backend.task_manager import TaskManager
ready, stop = Event(), Event()
def operation(log):
    ready.set()
    stop.wait(60)
    return {}
manager = TaskManager(Path(sys.argv[1]))
task = manager.submit('crash fixture', operation)
ready.wait(10)
(task.work_dir / 'preview.svg').write_bytes(b'keep preview')
print(task.id, flush=True)
sys.stdin.read()
"""
        process = subprocess.Popen([sys.executable, "-B", "-c", code, str(self.root / "tasks")],
                                   cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            task_id = process.stdout.readline().strip()
            self.assertRegex(task_id, r"^[a-f0-9]{32}$")
            reader = self.manager()
            self.assertEqual(reader.get(task_id)["status"], "running")
            original = (self.root / "tasks" / task_id / "task.json").read_bytes()
            process.terminate()
            process.wait(timeout=10)
            recovered = reader.get(task_id)
            self.assertEqual(recovered["status"], "failed")
            self.assertIn("已中断", recovered["error"])
            self.assertEqual((self.root / "tasks" / task_id / "preview.svg").read_bytes(), b"keep preview")
            self.assertEqual((self.root / "tasks" / task_id / "task.json").read_bytes(), original)
        finally:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)

    def test_shutdown_cancels_queue_and_keeps_running_work_directory(self):
        ready, release, queued_ran = Event(), Event(), Event()
        def operation(log):
            ready.set()
            release.wait(10)
            return {"done": True}
        manager = self.manager()
        active = manager.submit("active", operation)
        try:
            self.assertTrue(ready.wait(5))
            queued = manager.submit("queued", lambda log: queued_ran.set() or {})
            manager.shutdown()
            self.assertEqual(manager.get(queued.id)["status"], "cancelled")
            self.assertEqual(manager.get(active.id)["status"], "running")
            self.assertTrue(active.work_dir.is_dir())
            with self.assertRaises(TaskStateUnavailable):
                manager.submit("late", lambda log: {})
        finally:
            release.set()
        self.assertEqual(self.wait(manager, active.id)["status"], "succeeded")
        self.assertFalse(queued_ran.is_set())
        self.assertEqual(self.manager().get(queued.id)["status"], "cancelled")

    def test_corrupt_record_is_503_and_missing_record_is_404(self):
        manager = self.manager()
        created = manager.submit("saved", lambda log: {})
        self.wait(manager, created.id)
        record = created.work_dir / "task.json"
        reader = self.manager()
        with patch("web_backend.app.task_manager", reader), TestClient(app) as client:
            for raw in ("{", "[]", "null", '{"format_version":1,"task":{}}'):
                with self.subTest(raw=raw):
                    record.write_text(raw, encoding="utf-8")
                    self.assertEqual(client.get(f"/api/tasks/{created.id}").status_code, 503)
                    self.assertEqual(record.read_text(encoding="utf-8"), raw)
            self.assertEqual(client.get("/api/tasks/" + "f" * 32).status_code, 404)
            record.write_bytes(b"\xff\xfe")
            self.assertEqual(client.get(f"/api/tasks/{created.id}").status_code, 503)
            self.assertEqual(record.read_bytes(), b"\xff\xfe")

    def test_worker_exit_releases_ownership_and_persists_failure(self):
        def operation(log):
            raise SystemExit("worker exited")
        manager = self.manager()
        created = manager.submit("worker exit", operation)
        self.assertEqual(self.wait(manager, created.id)["status"], "failed")
        self.assertIsNone(created._lease)
        self.assertEqual(self.manager().get(created.id)["error"], "worker exited")

    def test_completion_during_recovery_is_not_misreported_as_interruption(self):
        manager = self.manager()
        created = manager.submit("finishing", lambda log: {"answer": 42})
        self.wait(manager, created.id)
        record = created.work_dir / "task.json"
        finished = record.read_bytes()
        running = json.loads(finished)
        running["task"]["status"] = "running"
        running["task"]["result"] = None
        record.write_text(json.dumps(running), encoding="utf-8")
        def complete_before_releasing_lease(directory):
            record.write_bytes(finished)
            return False
        with patch("platform_core.task_manager._owner_is_alive", side_effect=complete_before_releasing_lease):
            recovered = self.manager().get(created.id)
        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["result"], {"answer": 42})

    def test_initial_save_failure_does_not_start_the_operation(self):
        manager, ran = self.manager(), Event()
        with patch.object(manager, "_persist", side_effect=OSError("disk unavailable")):
            with self.assertRaises(TaskStateUnavailable):
                manager.submit("unrecorded", lambda log: ran.set() or {})
        self.assertFalse(ran.is_set())
        self.assertEqual(manager._tasks, {})

    def test_transient_save_failure_keeps_result_and_recovers_on_query(self):
        manager, ready, release = self.manager(), Event(), Event()
        def operation(log):
            ready.set()
            release.wait(10)
            return {"answer": 42}
        created = manager.submit("save retry", operation)
        try:
            self.assertTrue(ready.wait(5))
            record = created.work_dir / "task.json"
            before = record.read_bytes()
            with patch.object(manager, "_persist", side_effect=OSError("disk unavailable")):
                release.set()
                result = self.wait(manager, created.id)
                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(result["result"], {"answer": 42})
                self.assertIn("无法保存", result["persistence_error"])
                self.assertEqual(record.read_bytes(), before)
            self.assertIsNone(manager.get(created.id)["persistence_error"])
            self.assertEqual(self.manager().get(created.id)["result"], {"answer": 42})
        finally:
            release.set()


class SharedScoringTests(unittest.TestCase):
    def test_defaults_and_zero_weight_fallback(self) -> None:
        values = default_values()
        result = evaluate(values, {metric: 0 for metric in values})
        self.assertEqual(result["weighting_mode"], "equal_fallback")
        self.assertEqual(result["total_score"], 100.0)
        self.assertEqual(result["danger_score"], 0.0)


class LocalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    @patch("web_backend.app.path_picker.choose_local_path")
    def test_local_path_dialog_returns_absolute_path(self, choose_mock) -> None:
        selected_path = r"C:\example\Platform_sjtu\dnn_mpc\python_output"
        choose_mock.return_value = selected_path
        response = self.client.post(
            "/api/local-paths/select",
            json={"kind": "directory", "initial_path": selected_path},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"path": selected_path})
        choose_mock.assert_called_once_with("directory", selected_path)


    def test_health_scoring_and_sdg(self) -> None:
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        score = self.client.post("/api/score/evaluate", json={"values": default_values(), "weights": {}})
        self.assertEqual(score.status_code, 200)
        self.assertEqual(score.json()["total_score"], 100.0)

        example = self.client.get("/api/sdg/example")
        self.assertEqual(example.status_code, 200)
        editor_config = self.client.get("/api/sdg/config")
        self.assertEqual(editor_config.status_code, 200)
        self.assertEqual([item["label"] for item in editor_config.json()["fuzzy_terms"]], ["很小", "小", "较小", "中等", "较大", "大", "很大"])
        self.assertEqual(editor_config.json()["node_defaults"]["id"], "R1")
        analyzed = self.client.post("/api/sdg/analyze", json=example.json())
        self.assertEqual(analyzed.status_code, 200)
        analysis = analyzed.json()
        self.assertGreaterEqual(len(analysis["sil_recommendations"]), 1)
        self.assertEqual(analysis["sis_required_nodes"], [item["node_id"] for item in analysis["sil_recommendations"]])
        self.assertEqual(len(analysis["backward_paths"]), 2)
        self.assertTrue(all(item["paths"] for item in analysis["backward_paths"]))

    def test_sdg_request_rejects_invalid_boundaries_before_analysis(self) -> None:
        example = self.client.get("/api/sdg/example").json()
        invalid_payloads = []

        for value in (-1,):
            payload = deepcopy(example)
            payload["nodes"][0]["probability"] = value
            invalid_payloads.append(payload)
        for value in (-0.1, 1.1):
            payload = deepcopy(example)
            payload["edges"][0]["probability"] = value
            invalid_payloads.append(payload)

        empty_node_id = deepcopy(example)
        empty_node_id["nodes"][0]["id"] = "  "
        invalid_payloads.append(empty_node_id)
        empty_node_name = deepcopy(example)
        empty_node_name["nodes"][0]["name"] = "  "
        invalid_payloads.append(empty_node_name)
        empty_source = deepcopy(example)
        empty_source["edges"][0]["source"] = "  "
        invalid_payloads.append(empty_source)
        empty_target = deepcopy(example)
        empty_target["edges"][0]["target"] = "  "
        invalid_payloads.append(empty_target)

        with patch("web_backend.app.sdg.analyze") as analyze_mock:
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    response = self.client.post("/api/sdg/analyze", json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertNotIn("forward_paths", response.json())
            analyze_mock.assert_not_called()

        for value in (float("nan"), float("inf")):
            payload = deepcopy(example)
            payload["nodes"][0]["probability"] = value
            response = self.client.post(
                "/api/sdg/analyze",
                content=json.dumps(payload, allow_nan=True),
                headers={"content-type": "application/json"},
            )
            self.assertEqual(response.status_code, 422, response.text)

        with self.assertRaises(ValidationError):
            SDGNodeRequest(id="R1", name="cause", type="R", probability=float("nan"))
        with self.assertRaises(ValidationError):
            SDGNodeRequest(id="R1", name="cause", type="R", probability=float("inf"))

    def test_sdg_structural_errors_return_bad_request(self) -> None:
        example = self.client.get("/api/sdg/example").json()

        duplicate_nodes = deepcopy(example)
        duplicate_nodes["nodes"].append(deepcopy(duplicate_nodes["nodes"][0]))
        response = self.client.post("/api/sdg/analyze", json=duplicate_nodes)
        self.assertEqual(response.status_code, 400, response.text)

        missing_reference = deepcopy(example)
        missing_reference["edges"][0]["target"] = "missing-node"
        response = self.client.post("/api/sdg/analyze", json=missing_reference)
        self.assertEqual(response.status_code, 400, response.text)

        missing_cause = {"nodes": [
            {"id": "P1", "name": "parameter", "type": "P", "probability": 0},
            {"id": "C1", "name": "consequence", "type": "C", "probability": 0},
        ], "edges": []}
        response = self.client.post("/api/sdg/analyze", json=missing_cause)
        self.assertEqual(response.status_code, 400, response.text)


    def test_anomaly_request_accepts_inclusive_boundaries(self) -> None:
        request = AnomalyTaskRequest(
            mcr_root=r" E:\MATLAB2024 ",
            attack_min_pct=5,
            attack_max_pct=50,
            measurement_noise_pct=1,
            process_disturbance_pct=30,
        )
        self.assertEqual(request.mcr_root, r"E:\MATLAB2024")
        self.assertEqual(request.attack_min_pct, 5)
        self.assertEqual(request.attack_max_pct, 50)
        self.assertEqual(request.measurement_noise_pct, 1)
        self.assertEqual(request.process_disturbance_pct, 30)

    def test_anomaly_request_rejects_invalid_parameters_before_task_creation(self) -> None:
        invalid_payloads = [
            {"attack_min_pct": 4.9},
            {"attack_max_pct": 50.1},
            {"measurement_noise_pct": 0.9},
            {"process_disturbance_pct": 30.1},
            {"attack_min_pct": 20, "attack_max_pct": 10},
            {"mcr_root": "   "},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/anomaly/tasks", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("task_id", response.json())

        with self.assertRaises(ValidationError):
            AnomalyTaskRequest(attack_min_pct=float("nan"))




    def test_cdq_config_and_sample_first_analysis(self) -> None:
        config_response = self.client.get("/api/cdq/config")
        self.assertEqual(config_response.status_code, 200)
        config = config_response.json()
        self.assertEqual(len(config["u_labels"]), 7)
        self.assertEqual(len(config["cv_labels"]), 7)
        self.assertEqual(len(config["initial_u_now"]), 7)
        self.assertEqual(len(config["initial_u_after"]), 7)
        for vector_name in ("u_now", "u_after", "cv"):
            self.assertEqual(config["vector_bounds"][vector_name], [{"min": 0.0, "max": 999999.0}] * 7)
        self.assertEqual(config["vector_bounds_source"], "fixed")

        analyzed = self.client.post(
            "/api/cdq/analyze",
            json={
                "step": config["default_step"],
                "horizon": 3,
                "sample_index": 0,
                "cv": config["default_cv"],
                "u_now": config["initial_u_now"],
                "u_after": config["initial_u_after"],
            },
        )
        self.assertEqual(analyzed.status_code, 200, analyzed.text)
        result = analyzed.json()
        self.assertEqual(result["data_source"]["mode"], "dataset")
        self.assertEqual(result["data_source"]["sample_index"], 0)
        self.assertEqual(len(result["inputs"]["u_now"]), 7)
        self.assertEqual(len(result["series"]["steps"]), 3)
        self.assertEqual(len(result["risks"]), len(result["schemes"]))

    def test_cdq_vectors_non_negative_finite_bounds(self) -> None:
        config = self.client.get("/api/cdq/config").json()
        base = {
            "step": config["default_step"],
            "horizon": 3,
            "sample_index": 0,
            "cv": list(config["default_cv"]),
            "u_now": config["initial_u_now"],
            "u_after": config["initial_u_after"],
        }
        with patch("web_backend.app.cdq.analyze", return_value={"ok": True}) as analyze_mock:
            for value in (0.0, 999999.0):
                payload = deepcopy(base)
                payload["cv"][0] = value
                response = self.client.post("/api/cdq/analyze", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
            analyze_mock.assert_called()

        invalid_values = [-0.01, -1.0, 1000000.0]
        with patch("web_backend.app.cdq.analyze") as analyze_mock:
            for value in invalid_values:
                payload = deepcopy(base)
                payload["cv"][0] = value
                with self.subTest(value=value):
                    response = self.client.post("/api/cdq/analyze", json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
            analyze_mock.assert_not_called()

        from web_backend.app import CDQRequest
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    CDQRequest(cv=[value] + list(base["cv"])[1:])

    def test_cdq_control_action_boundaries_reject_out_of_range_values(self) -> None:
        config = self.client.get("/api/cdq/config").json()
        base = {
            "step": config["default_step"],
            "horizon": 3,
            "sample_index": 0,
            "cv": config["default_cv"],
            "u_now": config["initial_u_now"],
            "u_after": config["initial_u_after"],
        }
        for vector_name in ("u_now", "u_after"):
            for index, bound in enumerate(config["vector_bounds"][vector_name]):
                for value in (bound["min"] - 1e-6, bound["max"] + 1e-6):
                    payload = deepcopy(base)
                    payload[vector_name][index] = value
                    with self.subTest(vector_name=vector_name, index=index, value=value):
                        with patch("web_backend.app.cdq.analyze") as analyze_mock:
                            response = self.client.post("/api/cdq/analyze", json=payload)
                            self.assertEqual(response.status_code, 422, response.text)
                            analyze_mock.assert_not_called()

    def test_sil_request_accepts_defaults_and_rejects_invalid_boundaries(self) -> None:
        defaults_response = self.client.get("/api/sil/defaults")
        self.assertEqual(defaults_response.status_code, 200)
        request = SILTaskRequest(**defaults_response.json())
        self.assertEqual((request.m, request.n), (2, 4))
        self.assertEqual(request.total_beta, 0.1)
        self.assertEqual(SILTaskRequest(total_beta=1).total_beta, 1)

        invalid_payloads = [
            {"m": -1}, {"n": -1}, {"lambda_fit": -1}, {"ti": -1},
            {"mrt": -1}, {"nsim": -1}, {"years": -1},
            {"m": 11}, {"n": 11}, {"m": 4, "n": 2},
            {"total_beta": -0.01}, {"total_beta": 1.01},
            {"ccf_mode": "partial", "partial_betas": {"2": -0.01}},
            {"ccf_mode": "partial", "partial_betas": {"2": 1.01}},
            {"ccf_mode": "partial", "partial_betas": {"2": 0.6, "3": 0.4}},
        ]
        with patch("web_backend.app.task_manager.submit") as submit_mock:
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    response = self.client.post("/api/sil/tasks", json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertNotIn("task_id", response.json())
            submit_mock.assert_not_called()

        with self.assertRaises(ValidationError):
            SILTaskRequest(lambda_fit=float("nan"))
        with self.assertRaises(ValidationError):
            SILTaskRequest(ti=float("inf"))
        with self.assertRaises(ValidationError):
            SILTaskRequest(partial_betas={"2": float("nan")})

    def test_sil_task_reaches_a_terminal_state(self) -> None:
        created = self.client.post(
            "/api/sil/tasks",
            json={
                "m": 1,
                "n": 1,
                "lambda_fit": 111.11,
                "ti": 8760,
                "mrt": 8,
                "nsim": 1,
                "years": 1001,
                "ccf_mode": "total",
                "total_beta": 0.1,
            },
        )
        self.assertEqual(created.status_code, 200)
        task_id = created.json()["task_id"]
        # The first SIL request lazily imports Qt/matplotlib; allow a cold start.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            task = self.client.get(f"/api/tasks/{task_id}").json()
            if task["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertIn("pfdavg", task["result"])












if __name__ == "__main__":
    unittest.main()
