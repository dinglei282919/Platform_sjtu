"""Focused regression tests for the local B/S first-phase services."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from platform_core.scoring import default_values, evaluate
from web_backend.app import AnomalyTaskRequest, MpcTaskRequest, TrainingTaskRequest, app


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
        selected_path = r"H:\repository\Platform_sjtu\dnn_mpc\output"
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

    def test_training_defaults_and_existing_result_images(self) -> None:
        modules = self.client.get("/api/modules")
        self.assertEqual(modules.status_code, 200)
        training_module = next(item for item in modules.json() if item["id"] == "training")
        self.assertEqual(training_module["status"], "available")

        defaults = self.client.get("/api/training/defaults")
        self.assertEqual(defaults.status_code, 200)
        self.assertEqual(defaults.json()["package_name"], "dnnmpcpkg")
        self.assertEqual(defaults.json()["hidden_layers"], "64,64")

        for image_name in ("training_performance.png", "prediction_error.png"):
            image = self.client.get(f"/api/training/images/{image_name}")
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image.headers["content-type"], "image/png")

    def test_mpc_defaults_and_existing_result_images(self) -> None:
        modules = self.client.get("/api/modules")
        self.assertEqual(modules.status_code, 200)
        mpc_module = next(item for item in modules.json() if item["id"] == "dnn-mpc")
        self.assertEqual(mpc_module["status"], "available")

        defaults = self.client.get("/api/mpc/defaults")
        self.assertEqual(defaults.status_code, 200)
        self.assertEqual(defaults.json()["prediction_horizon"], 5)
        self.assertIn("process_control_nn_model.mat", defaults.json()["model_path"])

        for image_name in ("process_control_trajectory.png", "control_input.png", "tracking_error.png", "cost_curve.png"):
            image = self.client.get(f"/api/mpc/images/{image_name}")
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image.headers["content-type"], "image/png")

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
        minimum = TrainingTaskRequest(
            package_dir="",
            mcr_root=r"H:\software\matlab software\R2024b Runtime\R2024b",
            output_dir=r"H:\repository\Platform_sjtu\dnn_mpc\output",
            model_path=r"H:\repository\Platform_sjtu\dnn_mpc\output\model.mat",
            sample_count=100,
            epochs=1,
            hidden_layers="1",
        )
        maximum = TrainingTaskRequest(
            package_dir=r"H:\repository\Platform_sjtu\dnn_mpc\build_python",
            mcr_root=r"H:\software\matlab software\R2024b Runtime\R2024b",
            output_dir=r"H:\repository\Platform_sjtu\dnn_mpc\output",
            model_path=r"H:\repository\Platform_sjtu\dnn_mpc\output\model.mat",
            sample_count=100000,
            epochs=5000,
            hidden_layers=",".join(["4096"] * 10),
        )
        self.assertEqual(minimum.sample_count, 100)
        self.assertEqual(maximum.epochs, 5000)
        self.assertEqual(len(maximum.hidden_layers.split(",")), 10)

    def test_dnn_training_request_rejects_parameters_and_paths_before_task_creation(self) -> None:
        invalid_payloads = [
            {"sample_count": 99},
            {"sample_count": 100001},
            {"epochs": 0},
            {"epochs": 5001},
            {"hidden_layers": "64,0"},
            {"hidden_layers": ",".join(["64"] * 11)},
            {"package_name": "dnn-mpc"},
            {"package_dir": r"dnn_mpc\build_python"},
            {"mcr_root": r"relative\runtime"},
            {"output_dir": r"relative\output"},
            {"model_path": r"H:\repository\Platform_sjtu\dnn_mpc\output\model.txt"},
            {"dataset_path": r"relative\dataset.mat"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/training/tasks", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("task_id", response.json())

    def test_mpc_request_rejects_parameters_and_paths_before_task_creation(self) -> None:
        invalid_payloads = [
            {"sim_time": 0.19},
            {"sim_time": 20.01},
            {"prediction_horizon": 0},
            {"prediction_horizon": 61},
            {"package_name": "dnn-mpc"},
            {"mcr_root": r"relative\runtime"},
            {"output_dir": r"relative\output"},
            {"model_path": r"H:\repository\Platform_sjtu\dnn_mpc\output\model.txt"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/mpc/tasks", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("task_id", response.json())

        with self.assertRaises(ValidationError):
            MpcTaskRequest(sim_time=float("nan"))

    def test_cdq_config_and_sample_first_analysis(self) -> None:
        config_response = self.client.get("/api/cdq/config")
        self.assertEqual(config_response.status_code, 200)
        config = config_response.json()
        self.assertEqual(len(config["u_labels"]), 7)
        self.assertEqual(len(config["cv_labels"]), 7)
        self.assertEqual(len(config["initial_u_now"]), 7)
        self.assertEqual(len(config["initial_u_after"]), 7)

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
        for _ in range(40):
            task = self.client.get(f"/api/tasks/{task_id}").json()
            if task["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
        self.assertEqual(task["status"], "succeeded", task)
        self.assertIn("pfdavg", task["result"])


if __name__ == "__main__":
    unittest.main()
