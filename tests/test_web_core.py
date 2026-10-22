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
from web_backend.app import AnomalyTaskRequest, MpcTaskRequest, SDGNodeRequest, SILTaskRequest, TrainingTaskRequest, app


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
        image = artifacts / "training_rmse.svg"
        image.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        created = manager.submit("saved result", lambda log: {"answer": 42}, artifact_dir=artifacts)
        finished = self.wait(manager, created.id)
        original = (created.work_dir / "task.json").read_bytes()
        manager.shutdown()
        recovered = self.manager()
        self.assertEqual(recovered.get(created.id)["result"], finished["result"])
        self.assertEqual(recovered.artifact_dir(created.id), artifacts)
        with patch("web_backend.app.task_manager", recovered), TestClient(app) as client:
            self.assertEqual(client.get(f"/api/tasks/{created.id}").json()["status"], "succeeded")
            response = client.get(f"/api/training/tasks/{created.id}/images/training_rmse.svg")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, image.read_bytes())
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
        with patch("web_backend.task_manager._owner_is_alive", side_effect=complete_before_releasing_lease):
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

    def test_python_dnn_defaults_and_image_whitelist(self) -> None:
        modules = self.client.get("/api/modules").json()
        self.assertEqual(next(x for x in modules if x["id"] == "dnn-mpc")["status"], "available")
        defaults = self.client.get("/api/training/defaults").json()
        self.assertEqual(defaults["hidden_layers"], "64,64")
        self.assertEqual(defaults["epochs"], 200)
        self.assertEqual(defaults["lr"], .001)
        self.assertNotIn("mcr_root", defaults)
        self.assertTrue(defaults["model_path"].endswith(".npz"))
        defaults = self.client.get("/api/mpc/defaults").json()
        self.assertEqual(defaults["prediction_horizon"], 10)
        self.assertEqual(defaults["sim_time"], 5.)
        for url in ("/api/training/images/prediction_error.png", "/api/mpc/images/cost_curve.png",
                    "/api/mpc/tasks/missing/images/dnn_mpc.svg", "/api/dnn/training/tasks/missing/files/trained_model.npz"):
            self.assertEqual(self.client.get(url).status_code, 404)

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

    def test_dnn_training_request_accepts_inclusive_boundaries(self) -> None:
        minimum = TrainingTaskRequest(epochs=1, lr=.000001, hidden_layers="1,1")
        maximum = TrainingTaskRequest(epochs=5000, lr=.1, hidden_layers="512,512")
        self.assertEqual(minimum.epochs, 1)
        self.assertEqual(maximum.epochs, 5000)
        self.assertEqual(maximum.hidden_layers, "512,512")

    def test_dnn_training_request_rejects_parameters_and_paths_before_task_creation(self) -> None:
        invalid_payloads = [
            {"lr": 0.0000009},
            {"lr": .10001},
            {"epochs": 0},
            {"epochs": 5001},
            {"hidden_layers": "64,0"},
            {"hidden_layers": ",".join(["64"] * 11)},
            {"package_name": "dnn-mpc"},
            {"package_dir": r"unused\package"},
            {"mcr_root": r"relative\runtime"},
            {"output_dir": r"relative\output"},
            {"model_path": r"C:\example\Platform_sjtu\dnn_mpc\python_output\model.txt"},
            {"dataset_path": r"relative\dataset.mat"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/training/tasks", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("task_id", response.json())

        for field, value in (("lr", float("nan")), ("epochs", float("inf"))):
            with self.subTest(field=field):
                payload = {field: value}
                response = self.client.post(
                    "/api/training/tasks",
                    content=json.dumps(payload, allow_nan=True),
                    headers={"content-type": "application/json"},
                )
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("task_id", response.json())

        with self.assertRaises(ValidationError):
            TrainingTaskRequest(sample_count=-1)
        with self.assertRaises(ValidationError):
            TrainingTaskRequest(epochs=0)
        with self.assertRaises(ValidationError):
            TrainingTaskRequest(hidden_layers="64,0")

    def test_mpc_request_rejects_parameters_and_paths_before_task_creation(self) -> None:
        invalid_payloads = [
            {"sim_time": 0.19},
            {"sim_time": 20.01},
            {"prediction_horizon": 0},
            {"prediction_horizon": 61},
            {"package_name": "dnn-mpc"},
            {"mcr_root": r"relative\runtime"},
            {"output_dir": r"relative\output"},
            {"model_path": r"C:\example\Platform_sjtu\dnn_mpc\python_output\model.txt"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/mpc/tasks", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("task_id", response.json())

        with self.assertRaises(ValidationError):
            MpcTaskRequest(sim_time=float("nan"))
        with self.assertRaises(ValidationError):
            MpcTaskRequest(sim_time=0.1)
        with self.assertRaises(ValidationError):
            MpcTaskRequest(prediction_horizon=61)

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


def check_running_control_images(test, mode, *, cancel=False):
    """Pause a real run after its first preview and exercise live image routes."""
    from importlib import import_module
    from pathlib import Path
    from threading import Event
    from xml.etree import ElementTree
    ready, release = Event(), Event()
    runner = import_module(f"platform_core.risk_control.{mode}.runner")

    original_run = runner.run

    def gated_run(config, log, *, run_dir, cancelled=None):
        def progress(percent, message):
            state = json.loads((Path(run_dir) / "progress.json").read_text(encoding="utf-8"))
            if state["image_revision"] and percent < 85 and not release.is_set():
                ready.set()
                if not release.wait(20):
                    raise RuntimeError("preview API test gate timed out")
        return original_run(config, log, run_dir=run_dir, progress=progress, cancelled=cancelled)

    try:
        with patch.object(runner, "run", side_effect=gated_run):
            task_id = test.client.post(f"/api/{mode}/tasks", json={"sim_time": 1.}).json()["task_id"]
            test.assertTrue(ready.wait(20))
            task = test.manager.get(task_id)
            test.assertEqual(task["status"], "running")
            test.assertIsNone(task["result"])
            for name in task["progress"]["outputs"]:
                response = test.client.get(f"/api/{mode}/tasks/{task_id}/images/{name}.svg?v=1")
                test.assertEqual(response.status_code, 200)
                test.assertEqual(response.headers["cache-control"], "no-store")
                ElementTree.fromstring(response.content)
                test.assertEqual(test.client.get(f"/api/{mode}/tasks/{task_id}/files/{name}.svg").status_code, 409)
            test.assertEqual(test.client.get(f"/api/{mode}/tasks/{task_id}/files/result.json").status_code, 409)
            test.assertEqual(test.client.get(f"/api/{mode}/tasks/{task_id}/images/tracking.png").status_code, 409)
            other = "replay" if mode == "fdi" else "fdi"
            test.assertNotEqual(test.client.get(f"/api/{other}/tasks/{task_id}/images/tracking.svg").status_code, 200)
            if cancel:
                response = test.client.post(f"/api/tasks/{task_id}/cancel")
                test.assertEqual(response.status_code, 200)
                test.assertTrue(response.json()["cancel_requested"])
                test.assertEqual(response.json()["status"], "running")
            release.set()
            final = test.wait_task(task_id)
            if cancel:
                test.assertEqual(final["status"], "cancelled", final)
                test.assertIsNone(final["result"])
                test.assertIsNone(final["error"])
                test.assertEqual(test.client.get(f"/api/{mode}/tasks/{task_id}/files/result.json").status_code, 409)
                test.assertEqual(test.client.get(f"/api/{mode}/tasks/{task_id}/images/tracking.svg").status_code, 200)
                test.assertEqual(test.client.post(f"/api/tasks/{task_id}/cancel").json(), final)
            else:
                test.assertEqual(final["status"], "succeeded", final)
                test.assertGreater(final["progress"]["image_revision"], task["progress"]["image_revision"])
                test.assertEqual(test.client.post(f"/api/tasks/{task_id}/cancel").json(), final)
    finally:
        release.set()


class FdiApiTests(unittest.TestCase):
    def test_queued_cancellation_and_unsupported_tasks(self):
        from threading import Event
        started, release, cancelled = Event(), Event(), Event()
        invoked = []

        def occupy(log):
            started.set()
            if not release.wait(10):
                raise RuntimeError("queue test gate timed out")
            return {"finished": True}

        first = self.manager.submit("uncancellable", occupy)
        try:
            self.assertTrue(started.wait(5))
            second = self.manager.submit("queued", lambda log: invoked.append(True), cancel_event=cancelled)
            self.assertEqual(self.client.post("/api/tasks/missing/cancel").status_code, 404)
            self.assertEqual(self.client.post(f"/api/tasks/{first.id}/cancel").status_code, 409)
            response = self.client.post(f"/api/tasks/{second.id}/cancel")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "cancelled")
            self.assertTrue(cancelled.is_set())
            self.assertEqual(self.client.post(f"/api/tasks/{second.id}/cancel").json(), response.json())
        finally:
            release.set()
            self.manager._executor.shutdown(wait=True)
        self.assertEqual(invoked, [])
        self.assertEqual(self.manager.get(first.id)["status"], "succeeded")

    def test_running_cancellation_preserves_preview_but_not_final_results(self):
        check_running_control_images(self, "fdi", cancel=True)

    def test_running_images_are_published_before_final_exports(self):
        check_running_control_images(self, "fdi")

    def setUp(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from web_backend.task_manager import TaskManager
        from web_backend.app import app
        self.folder = TemporaryDirectory()
        self.manager = TaskManager(Path(self.folder.name) / "tasks")
        self.output_patch = patch("platform_core.risk_control.fdi.runner.DEFAULT_OUTPUT_DIR", Path(self.folder.name) / "outputs")
        self.output_patch.start()
        manager_patch = patch("web_backend.app.task_manager", self.manager)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.manager._executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.output_patch.stop()
        self.folder.cleanup()

    def wait_task(self, task_id):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            task = self.manager.get(task_id)
            if task["status"] in ("succeeded", "failed", "cancelled"):
                return task
            time.sleep(.05)
        self.fail("FDI task did not complete")

    def test_fdi_validation_and_missing_artifacts(self):
        self.assertEqual(self.client.get("/api/fdi/defaults").json(), {"sim_time": 10.})
        for payload in ({"sim_time": 0}, {"sim_time": 11}, {"sim_time": True},
                        {"sim_time": "10"}, {"unexpected": 1}):
            self.assertEqual(self.client.post("/api/fdi/tasks", json=payload).status_code, 422)
        self.assertEqual(self.client.get("/api/fdi/tasks/missing/images/tracking.svg").status_code, 404)
        from web_backend.app import FdiTaskRequest
        with self.assertRaises(ValidationError):
            FdiTaskRequest(sim_time=float("nan"))
        with self.assertRaises(ValidationError):
            FdiTaskRequest(sim_time=float("inf"))

    def test_fdi_task_images_downloads_and_failure(self):
        created = self.client.post("/api/fdi/tasks", json={"sim_time": .2})
        self.assertEqual(created.status_code, 200)
        task_id = created.json()["task_id"]
        task = self.wait_task(task_id)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertEqual(task["progress"]["percent"], 100)
        for name in ("tracking", "disturbance", "state_control", "gain_switching"):
            response = self.client.get(f"/api/fdi/tasks/{task_id}/images/{name}.svg")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("image/svg+xml"))
            self.assertIn("<svg", response.text)
        for name in ("signals.csv", "signals.npz", "result.json"):
            response = self.client.get(f"/api/fdi/tasks/{task_id}/files/{name}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(self.client.get(f"/api/fdi/tasks/{task_id}/files/fdi_config.json").status_code, 404)
        with patch("platform_core.risk_control.fdi.runner.run", side_effect=RuntimeError("expected FDI failure")):
            failed_id = self.client.post("/api/fdi/tasks", json={}).json()["task_id"]
            failed = self.wait_task(failed_id)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("expected FDI failure", failed["error"])
        self.assertEqual(self.client.get(f"/api/fdi/tasks/{failed_id}/images/tracking.svg").status_code, 409)


class ReplayApiTests(unittest.TestCase):
    def test_running_cancellation_preserves_preview_but_not_final_results(self):
        check_running_control_images(self, "replay", cancel=True)

    def test_running_images_are_published_before_final_exports(self):
        check_running_control_images(self, "replay")

    def setUp(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from web_backend.task_manager import TaskManager
        from web_backend.app import app
        self.folder = TemporaryDirectory()
        self.manager = TaskManager(Path(self.folder.name) / "tasks")
        self.output_patch = patch("platform_core.risk_control.replay.runner.DEFAULT_OUTPUT_DIR", Path(self.folder.name) / "outputs")
        self.output_patch.start()
        manager_patch = patch("web_backend.app.task_manager", self.manager)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.manager._executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.output_patch.stop()
        self.folder.cleanup()

    def wait_task(self, task_id):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            task = self.manager.get(task_id)
            if task["status"] in ("succeeded", "failed", "cancelled"):
                return task
            time.sleep(.05)
        self.fail("replay task did not complete")

    def test_replay_validation_and_missing_artifacts(self):
        self.assertEqual(self.client.get("/api/replay/defaults").json(), {"sim_time": 50.})
        for payload in ({"sim_time": 0}, {"sim_time": 51}, {"sim_time": True},
                        {"sim_time": "10"}, {"unexpected": 1}):
            self.assertEqual(self.client.post("/api/replay/tasks", json=payload).status_code, 422)
        self.assertEqual(self.client.get("/api/replay/tasks/missing/images/tracking.svg").status_code, 404)
        from web_backend.app import ReplayTaskRequest
        with self.assertRaises(ValidationError):
            ReplayTaskRequest(sim_time=float("nan"))
        with self.assertRaises(ValidationError):
            ReplayTaskRequest(sim_time=float("inf"))

    def test_replay_task_images_downloads_and_failure(self):
        created = self.client.post("/api/replay/tasks", json={"sim_time": .2})
        self.assertEqual(created.status_code, 200)
        task_id = created.json()["task_id"]
        task = self.wait_task(task_id)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertEqual(task["progress"]["percent"], 100)
        for name in ("tracking", "disturbance", "state_control", "switching_replay"):
            response = self.client.get(f"/api/replay/tasks/{task_id}/images/{name}.svg")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("image/svg+xml"))
            self.assertIn("<svg", response.text)
        for name in ("signals.csv", "signals.npz", "result.json"):
            response = self.client.get(f"/api/replay/tasks/{task_id}/files/{name}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(self.client.get(f"/api/replay/tasks/{task_id}/files/replay_config.json").status_code, 404)
        with patch("platform_core.risk_control.replay.runner.run", side_effect=RuntimeError("expected replay failure")):
            failed_id = self.client.post("/api/replay/tasks", json={}).json()["task_id"]
            failed = self.wait_task(failed_id)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("expected replay failure", failed["error"])
        self.assertEqual(self.client.get(f"/api/replay/tasks/{failed_id}/images/tracking.svg").status_code, 409)




class SelfHealingApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = TaskManager(self.root / "tasks")
        self.manager_patch = patch("web_backend.app.task_manager", self.manager)
        self.output_patch = patch("platform_core.risk_control.self_healing.runner.DEFAULT_OUTPUT_DIR", self.root / "outputs")
        self.manager_patch.start(); self.output_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.manager.shutdown()
        self.manager._executor.shutdown(wait=True)
        self.manager_patch.stop(); self.output_patch.stop()
        self.client.close()
        self.temp.cleanup()

    def wait_task(self, task_id):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/tasks/{task_id}")
            self.assertEqual(response.status_code, 200)
            task = response.json()
            if task["status"] in {"succeeded", "failed", "cancelled"}:
                return task
            time.sleep(.02)
        self.fail("Self-healing task did not terminate")

    def config(self):
        from platform_core.risk_control.self_healing.config import defaults
        return {**defaults(), "duration": 1., "fault_start": .3, "training_batches": 8, "training_hold": .2}

    def test_defaults_validation_and_module_isolation(self):
        from platform_core.risk_control.self_healing.config import catalog
        self.assertEqual(self.client.get("/api/self-healing/defaults").json(), catalog())
        modules = self.client.get("/api/modules").json()
        self.assertTrue(any(module["id"] == "self-healing" for module in modules))
        for config in ({"duration": True}, {"training_batches": 1}, {"fault_start": 1000},
                       {"dt": .003}, {"bad": 1}, {"scenario": "missing"}, {"seed": 10**400}):
            with self.subTest(config=config):
                response = self.client.post("/api/self-healing/tasks", json={"config": config})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse((self.root / "outputs").exists())
        for path in ("files/summary.json", "images/outputs.svg", "files/self_healing_config.json"):
            self.assertEqual(self.client.get("/api/self-healing/tasks/missing/" + path).status_code, 404)

    def test_result_downloads_progress_restart_and_cross_module_denial(self):
        from platform_core.risk_control.self_healing.runner import ARTIFACT_NAMES
        created = self.client.post("/api/self-healing/tasks", json={"config": self.config()})
        self.assertEqual(created.status_code, 200, created.text)
        task_id = created.json()["task_id"]
        task = self.wait_task(task_id)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertEqual(task["progress"]["percent"], 100)
        for name in ARTIFACT_NAMES:
            response = self.client.get(f"/api/self-healing/tasks/{task_id}/files/{name}")
            self.assertEqual(response.status_code, 200, name)
        for name in ("outputs.svg", "inputs.png"):
            self.assertEqual(self.client.get(f"/api/self-healing/tasks/{task_id}/images/{name}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/replay/tasks/{task_id}/files/result.json").status_code, 404)
        recovered = TaskManager(self.root / "tasks")
        try:
            with patch("web_backend.app.task_manager", recovered):
                response = self.client.get(f"/api/self-healing/tasks/{task_id}/files/result.json")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["config"], self.config())
        finally:
            recovered.shutdown(); recovered._executor.shutdown(wait=True)

    def test_worker_failure_and_cancellation_allow_retry(self):
        with patch("platform_core.risk_control.self_healing.runner.run_experiment", side_effect=RuntimeError("test solver failure")):
            task_id = self.client.post("/api/self-healing/tasks", json={"config": self.config()}).json()["task_id"]
            task = self.wait_task(task_id)
        self.assertEqual(task["status"], "failed")
        self.assertIn("test solver failure", task["error"])
        self.assertEqual(self.client.get(f"/api/self-healing/tasks/{task_id}/files/result.json").status_code, 409)
        from platform_core.risk_control.self_healing.simulation import SimulationCancelled
        ready = Event()
        def cancellable(*args, cancelled, **kwargs):
            ready.set()
            deadline = time.monotonic() + 5
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(.01)
            raise SimulationCancelled("cancelled")
        with patch("platform_core.risk_control.self_healing.runner.run_experiment", side_effect=cancellable):
            task_id = self.client.post("/api/self-healing/tasks", json={"config": self.config()}).json()["task_id"]
            self.assertTrue(ready.wait(5))
            self.assertEqual(self.client.post(f"/api/tasks/{task_id}/cancel").status_code, 200)
            self.assertEqual(self.wait_task(task_id)["status"], "cancelled")
        task_id = self.client.post("/api/self-healing/tasks", json={"config": self.config()}).json()["task_id"]
        self.assertEqual(self.wait_task(task_id)["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
