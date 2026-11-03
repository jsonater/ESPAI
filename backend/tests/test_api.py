import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from fastapi.testclient import TestClient

import backend.main as api


class ConceptMeshApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        success, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
        if not success:
            raise RuntimeError("Could not create JPEG test frame")
        cls.jpeg = encoded.tobytes()

    def setUp(self):
        with api.CAMERA_STATUS_LOCK:
            api.CAMERA_STATUS.update(
                state="idle",
                width=None,
                height=None,
                fps=0.0,
                frames_received=0,
                last_frame_at=None,
                error=None,
            )
            api.CAMERA_FRAME_TIMES.clear()

    def test_health_route(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_root_serves_local_dashboard(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Training image library", response.text)

    def test_roboflow_status_does_not_expose_api_key(self):
        with patch.object(api, "ROBOFLOW_API_KEY", "private-test-key"), patch.object(api, "ROBOFLOW_MODEL_ID", "heritage-project/1"):
            response = self.client.get("/api/v1/roboflow/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "configured": True,
            "model_id": "heritage-project/1",
            "runtime": "serverless-cloud",
        })
        self.assertNotIn("private-test-key", response.text)

    def test_roboflow_prediction_requires_configuration(self):
        with patch.object(api, "ROBOFLOW_API_KEY", None), patch.object(api, "ROBOFLOW_MODEL_ID", None):
            response = self.client.post(
                "/api/v1/roboflow/predict",
                files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 503)

    def test_roboflow_prediction_uses_rgb_frame_and_returns_result(self):
        result = {"predictions": [{"class": "fracture", "confidence": 0.91}]}
        with (
            patch.object(api, "ROBOFLOW_API_KEY", "private-test-key"),
            patch.object(api, "ROBOFLOW_MODEL_ID", "heritage-project/1"),
            patch("backend.main.run_roboflow_inference", return_value=result) as inference,
        ):
            response = self.client.post(
                "/api/v1/roboflow/predict",
                files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)
        self.assertEqual(inference.call_args.args[0].shape, (24, 32, 3))

    def test_roboflow_client_uses_bearer_header_and_configured_model(self):
        image_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        expected = {"predictions": []}
        with (
            patch.object(api, "ROBOFLOW_API_KEY", "private-test-key"),
            patch.object(api, "ROBOFLOW_MODEL_ID", "heritage-project/1"),
            patch("backend.main.InferenceHTTPClient") as client_class,
            patch("backend.main.InferenceConfiguration") as configuration_class,
        ):
            client_class.return_value.configure.return_value.infer.return_value = expected

            result = api.run_roboflow_inference(image_rgb)

        client_class.assert_called_once_with(
            api_url=api.ROBOFLOW_API_URL,
            api_key="private-test-key",
        )
        configuration_class.assert_called_once_with(api_key_transport="header")
        client_class.return_value.configure.return_value.infer.assert_called_once_with(
            image_rgb,
            model_id="heritage-project/1",
        )
        self.assertEqual(result, expected)

    def test_camera_stream_returns_mjpeg_and_updates_status(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        with patch("backend.main.iter_camera_frames", return_value=iter([frame])):
            response = self.client.get("/api/v1/camera/stream")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("multipart/x-mixed-replace"))
        self.assertIn(b"--frame\r\n", response.content)
        self.assertIn(b"Content-Type: image/jpeg", response.content)
        status = self.client.get("/api/v1/camera/status").json()
        self.assertEqual(status["state"], "stopped")
        self.assertEqual((status["width"], status["height"]), (32, 24))
        self.assertEqual(status["frames_received"], 1)

    def test_camera_stream_reports_connection_failure(self):
        with patch("backend.main.iter_camera_frames", side_effect=RuntimeError("camera offline")):
            response = self.client.get("/api/v1/camera/stream")

        self.assertEqual(response.status_code, 200)
        status = self.client.get("/api/v1/camera/status").json()
        self.assertEqual(status["state"], "error")
        self.assertEqual(status["error"], "camera offline")

    def test_valid_camera_frame_returns_stl(self):
        response = self.client.post(
            "/api/v1/concept-mesh",
            files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
            data={"width_mm": "90", "height_mm": "42", "depth_mm": "36"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "model/stl")
        self.assertEqual(response.headers["x-image-resolution"], "32x24")
        mesh = trimesh.load(io.BytesIO(response.content), file_type="stl", force="mesh")
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(mesh.extents.tolist(), [90, 42, 36])

    def test_rejects_bad_image_and_invalid_dimensions(self):
        bad_image = self.client.post(
            "/api/v1/concept-mesh",
            files={"image": ("frame.jpg", b"not an image", "image/jpeg")},
            data={"width_mm": "90", "height_mm": "42", "depth_mm": "36"},
        )
        bad_dimensions = self.client.post(
            "/api/v1/concept-mesh",
            files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
            data={"width_mm": "0", "height_mm": "42", "depth_mm": "36"},
        )

        self.assertEqual(bad_image.status_code, 400)
        self.assertEqual(bad_dimensions.status_code, 422)


if __name__ == "__main__":
    unittest.main()