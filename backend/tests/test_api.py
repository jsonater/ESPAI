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