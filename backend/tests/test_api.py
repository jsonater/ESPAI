import base64
import asyncio
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from fastapi.testclient import TestClient

import backend.main as api


class FakeCameraWebSocket:
    def __init__(self, jpeg, event, *, fail=False):
        self.jpeg = jpeg
        self.event = event
        self.fail = fail
        self.sent_frame = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.fail:
            self.event.set()
            raise OSError("camera offline")
        if self.sent_frame:
            await asyncio.Future()
        self.sent_frame = True
        self.event.set()
        return self.jpeg


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
            api.CAMERA_LATEST_JPEG = None
            api.CAMERA_FRAME_NUMBER = 0
            api.CAMERA_LAST_FRAME_MONOTONIC = None

    def test_health_route(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_rebel_chat_sends_message_and_safe_diagnostics_to_ollama(self):
        response_mock = unittest.mock.MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.read.return_value = json.dumps({"response": "Check the car Wi-Fi."}).encode()
        with patch.object(api, "REBEL_MODEL", "support-test-model"), patch("backend.main.urlopen", return_value=response_mock) as request_mock:
            response = self.client.post("/api/v1/rebel/chat", json={"message": "Help with labeling images"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "Check the car Wi-Fi.")
        payload = json.loads(request_mock.call_args.args[0].data)
        self.assertEqual(payload["model"], "support-test-model")
        self.assertEqual(response.json()["model"], "support-test-model")
        self.assertEqual(json.loads(payload["prompt"])["question"], "Help with labeling images")
        self.assertIn("camera_state", payload["system"])
        self.assertNotIn("ROBOFLOW_API_KEY", payload["system"])

    def test_rebel_known_stl_help_uses_verified_steps_without_model(self):
        with patch("backend.main.urlopen", side_effect=AssertionError("Known help must not call the model")):
            response = self.client.post("/api/v1/rebel/chat", json={
                "message": "My GLB conversion failed",
                "conversion_status": "Choose a GLB file first",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "local-guide")
        self.assertIn("Open photo generator", response.json()["reply"])
        self.assertIn("Choose a GLB file first", response.json()["reply"])

    def test_rebel_guides_match_whole_words_only(self):
        self.assertIsNone(api.rebel_help_guide("How do I install Ollama?", ""))
        self.assertIsNone(api.rebel_help_guide("Why is my laptop slow?", ""))
        self.assertIn("Open photo generator", api.rebel_help_guide("my stl file is broken", ""))

    def test_rebel_whitefill_help_sets_scope_and_archive_help_is_accurate(self):
        whitefill = self.client.post("/api/v1/rebel/chat", json={"message": "Why can't Whitefill reconstruct a bookshelf?"})
        archive = self.client.post("/api/v1/rebel/chat", json={"message": "Where are my saved STL files?"})
        self.assertIn("pure-white pieces only", whitefill.json()["reply"])
        self.assertIn("not automatically saved", archive.json()["reply"])

    def test_rebel_chat_handles_unavailable_ollama_and_invalid_messages(self):
        self.assertEqual(self.client.post("/api/v1/rebel/chat", json={"message": " "}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/rebel/chat", json={"message": "x" * 4001}).status_code, 422)
        with patch("backend.main.urlopen", side_effect=OSError("offline")):
            response = self.client.post("/api/v1/rebel/chat", json={"message": "Help"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Ollama", response.json()["detail"])

    def test_root_serves_local_dashboard(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Training image library", response.text)

    def test_camera_capture_returns_a_valid_local_jpeg(self):
        response_mock = unittest.mock.MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.read.return_value = self.jpeg
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(api, "CAMERA_CAPTURE_URL", "http://192.168.4.1/capture"),
                patch.object(api, "CAPTURE_PHOTO_DIR", Path(temp_dir) / "ESP car photos"),
                patch("backend.main.urlopen", return_value=response_mock) as camera_request,
            ):
                response = self.client.get("/api/v1/camera/capture")

            output_dir = Path(temp_dir) / "ESP car photos"
            saved_photos = list(output_dir.iterdir())
            self.assertEqual(len(saved_photos), 1)
            self.assertEqual(saved_photos[0].suffix, ".jpg")
            self.assertEqual(saved_photos[0].read_bytes(), self.jpeg)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-image-resolution"], "32x24")
        self.assertTrue(response.content.startswith(b"\xff\xd8"))
        self.assertTrue(response.headers["x-saved-photo"].endswith(".jpg"))
        camera_request.assert_called_once_with("http://192.168.4.1/capture", timeout=5)

    def test_camera_capture_returns_latest_websocket_frame_when_streaming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "ESP car photos"
            with (
                patch.object(api, "CAMERA_LATEST_JPEG", self.jpeg),
                patch.object(api, "CAMERA_LAST_FRAME_MONOTONIC", time.monotonic()),
                patch.object(api, "CAMERA_STATUS", {**api.CAMERA_STATUS, "state": "streaming", "width": 32, "height": 24}),
                patch.object(api, "CAPTURE_PHOTO_DIR", output_dir),
                patch("backend.main.urlopen") as camera_request,
            ):
                response = self.client.get("/api/v1/camera/capture")

            saved_photos = list(output_dir.iterdir())
            self.assertEqual(len(saved_photos), 1)
            self.assertEqual(saved_photos[0].suffix, ".jpg")
            self.assertEqual(saved_photos[0].read_bytes(), self.jpeg)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-image-resolution"], "32x24")
        self.assertTrue(response.headers["x-saved-photo"].endswith(".jpg"))
        self.assertEqual(response.content, self.jpeg)
        camera_request.assert_not_called()

    def test_latest_frame_route_returns_cached_jpeg_and_frame_number(self):
        with (
            patch.object(api, "CAMERA_LATEST_JPEG", self.jpeg),
            patch.object(api, "CAMERA_FRAME_NUMBER", 42),
            patch.object(api, "CAMERA_LAST_FRAME_MONOTONIC", time.monotonic()),
            patch.object(api, "CAMERA_STATUS", {**api.CAMERA_STATUS, "state": "streaming"}),
        ):
            response = self.client.get("/api/v1/camera/frame")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-frame-number"], "42")
        self.assertEqual(response.content, self.jpeg)

    def test_latest_frame_route_returns_no_content_before_first_frame(self):
        response = self.client.get("/api/v1/camera/frame")

        self.assertEqual(response.status_code, 204)

    def test_latest_frame_route_hides_cached_frame_after_camera_disconnects(self):
        disconnected_status = {**api.CAMERA_STATUS, "state": "reconnecting"}
        with (
            patch.object(api, "CAMERA_LATEST_JPEG", self.jpeg),
            patch.object(api, "CAMERA_FRAME_NUMBER", 42),
            patch.object(api, "CAMERA_STATUS", disconnected_status),
        ):
            response = self.client.get("/api/v1/camera/frame")

        self.assertEqual(response.status_code, 204)

    def test_latest_frame_route_hides_stale_frame_when_socket_stalls(self):
        with (
            patch.object(api, "CAMERA_LATEST_JPEG", self.jpeg),
            patch.object(api, "CAMERA_LAST_FRAME_MONOTONIC", time.monotonic() - 10),
            patch.object(api, "CAMERA_STATUS", {**api.CAMERA_STATUS, "state": "streaming"}),
        ):
            response = self.client.get("/api/v1/camera/frame")

        self.assertEqual(response.status_code, 204)

    def test_camera_status_does_not_report_stale_fps_as_streaming(self):
        with (
            patch.object(api, "CAMERA_LATEST_JPEG", self.jpeg),
            patch.object(api, "CAMERA_LAST_FRAME_MONOTONIC", time.monotonic() - 10),
            patch.object(api, "CAMERA_STATUS", {**api.CAMERA_STATUS, "state": "streaming", "fps": 22.2}),
        ):
            status = self.client.get("/api/v1/camera/status").json()

        self.assertEqual(status["state"], "reconnecting")
        self.assertEqual(status["fps"], 0.0)

    def test_roboflow_status_does_not_expose_api_key(self):
        with patch.object(api, "ROBOFLOW_API_KEY", "private-test-key"), patch.object(api, "ROBOFLOW_MODEL_ID", "heritage-project/1"):
            response = self.client.get("/api/v1/roboflow/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "configured": True,
            "model_id": "heritage-project/1",
            "runtime": "serverless-cloud",
            "internet_required": True,
        })
        self.assertNotIn("private-test-key", response.text)

    def test_ollama_status_reports_installed_local_model(self):
        response_mock = unittest.mock.MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.read.return_value = json.dumps({"models": [{"name": "qwen3-vl:2b"}]}).encode()
        with patch("backend.main.urlopen", return_value=response_mock) as ollama_request:
            response = self.client.get("/api/v1/ollama/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "available": True,
            "model_installed": True,
            "model": "qwen3-vl:2b",
            "runtime": "local",
            "message": "Ready",
        })
        self.assertIn("127.0.0.1:11434/api/tags", ollama_request.call_args.args[0].full_url)

    def test_ollama_status_reports_when_service_is_not_running(self):
        with patch("backend.main.urlopen", side_effect=OSError("connection refused")):
            response = self.client.get("/api/v1/ollama/status")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["available"])
        self.assertFalse(response.json()["model_installed"])

    def test_ollama_inference_sends_local_image_and_parses_text_result(self):
        observation = "Fracture: a visible crack crosses the stone surface."
        response_mock = unittest.mock.MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.read.return_value = json.dumps({"response": observation, "done_reason": "stop"}).encode()
        with patch("backend.main.urlopen", return_value=response_mock) as ollama_request:
            result = api.run_ollama_inference(self.jpeg)

        request = ollama_request.call_args.args[0]
        request_payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(request_payload["model"], "qwen3-vl:2b")
        self.assertEqual(request_payload["images"], [base64.b64encode(self.jpeg).decode("ascii")])
        self.assertFalse(request_payload["stream"])
        self.assertFalse(request_payload["think"])
        self.assertNotIn("format", request_payload)
        self.assertEqual(request_payload["options"]["num_predict"], 1024)
        self.assertEqual(result["analysis"]["condition"], "Fracture")
        self.assertEqual(result["analysis"]["observations"], observation)
        self.assertIn("no physical scale", result["analysis"]["limitations"])
        self.assertEqual(result["provider"], "ollama")

    def test_ollama_inference_rejects_empty_truncated_output(self):
        response_mock = unittest.mock.MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.read.return_value = json.dumps({"response": "", "done_reason": "length"}).encode()
        with patch("backend.main.urlopen", return_value=response_mock):
            with self.assertRaisesRegex(api.OllamaInferenceError, "output limit"):
                api.run_ollama_inference(self.jpeg)

    def test_ollama_inference_does_not_treat_negated_damage_as_a_label(self):
        response_mock = unittest.mock.MagicMock()
        response_mock.__enter__.return_value = response_mock
        response_mock.read.return_value = json.dumps({
            "response": "No visible fracture is confirmed; uncertain.",
            "done_reason": "stop",
        }).encode()
        with patch("backend.main.urlopen", return_value=response_mock):
            result = api.run_ollama_inference(self.jpeg)

        self.assertEqual(result["analysis"]["condition"], "Uncertain")

    def test_ollama_prediction_rejects_invalid_images_without_calling_model(self):
        with patch("backend.main.run_ollama_inference") as inference:
            response = self.client.post(
                "/api/v1/ollama/predict",
                files={"image": ("frame.jpg", b"not an image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 400)
        inference.assert_not_called()

    def test_ollama_prediction_reports_unavailable_service(self):
        with patch("backend.main.run_ollama_inference", side_effect=api.OllamaUnavailableError("Ollama is offline")):
            response = self.client.post(
                "/api/v1/ollama/predict",
                files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("pull qwen3-vl:2b", response.json()["detail"].lower())

    def test_ollama_prediction_reports_slow_inference_as_timeout(self):
        with patch("backend.main.run_ollama_inference", side_effect=api.OllamaTimeoutError("Local model is slow")):
            response = self.client.post(
                "/api/v1/ollama/predict",
                files={"image": ("frame.jpg", self.jpeg, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"], "Local model is slow")

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

    def test_startup_receiver_caches_binary_jpeg_from_camera_websocket(self):
        frame_received = asyncio.Event()
        websocket = FakeCameraWebSocket(self.jpeg, frame_received)

        async def run_receiver_once():
            with (
                patch.object(api, "CAMERA_WEBSOCKET_URL", "ws://192.168.4.1/Camera"),
                patch("backend.main.websockets.connect", return_value=websocket),
            ):
                async with api.lifespan(api.app):
                    await asyncio.wait_for(frame_received.wait(), timeout=1)

        asyncio.run(run_receiver_once())

        status = self.client.get("/api/v1/camera/status").json()
        self.assertEqual(status["state"], "streaming")
        self.assertEqual(status["source"], "ws://192.168.4.1/Camera")
        self.assertEqual((status["width"], status["height"]), (32, 24))
        self.assertEqual(status["frames_received"], 1)
        self.assertEqual(api.CAMERA_LATEST_JPEG, self.jpeg)

    def test_receiver_reports_websocket_connection_failure(self):
        failed = asyncio.Event()
        websocket = FakeCameraWebSocket(self.jpeg, failed, fail=True)

        async def run_receiver_once():
            with patch("backend.main.websockets.connect", return_value=websocket):
                task = asyncio.create_task(api.receive_from_esp32())
                await asyncio.wait_for(failed.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(run_receiver_once())
        self.assertEqual(api.CAMERA_STATUS["state"], "reconnecting")
        self.assertEqual(api.CAMERA_STATUS["error"], "camera offline")

    def test_receiver_reconnects_when_frames_stop(self):
        async def run_receiver_once():
            frame_received = asyncio.Event()
            websocket = FakeCameraWebSocket(self.jpeg, frame_received)
            with (
                patch.object(api, "CAMERA_RECEIVE_TIMEOUT_SECONDS", 0.02),
                patch("backend.main.websockets.connect", return_value=websocket),
            ):
                task = asyncio.create_task(api.receive_from_esp32())
                await asyncio.wait_for(frame_received.wait(), timeout=1)
                await asyncio.sleep(0.05)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(run_receiver_once())
        self.assertEqual(api.CAMERA_STATUS["state"], "reconnecting")
        self.assertIn("No camera frames received", api.CAMERA_STATUS["error"])

    def test_receiver_timeout_explains_network_and_path_checks(self):
        async def run_receiver_once():
            with patch("backend.main.websockets.connect", side_effect=TimeoutError("opening handshake timed out")):
                task = asyncio.create_task(api.receive_from_esp32())
                for _ in range(10):
                    await asyncio.sleep(0)
                    if api.CAMERA_STATUS["state"] == "reconnecting":
                        break
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(run_receiver_once())
        self.assertIn("Wi-Fi is MyWiFiCar", api.CAMERA_STATUS["error"])
        self.assertIn("/Camera", api.CAMERA_STATUS["error"])

    def test_camera_stream_mjpeg_fans_out_cached_websocket_frame(self):
        async def read_one_chunk():
            response = api.camera_stream()
            iterator = response.body_iterator
            try:
                return await asyncio.wait_for(anext(iterator), timeout=1)
            finally:
                await iterator.aclose()

        with (
            patch.object(api, "CAMERA_LATEST_JPEG", self.jpeg),
            patch.object(api, "CAMERA_FRAME_NUMBER", 1),
            patch.object(api, "CAMERA_LAST_FRAME_MONOTONIC", time.monotonic()),
            patch.object(api, "CAMERA_STATUS", {**api.CAMERA_STATUS, "state": "streaming"}),
        ):
            response = api.camera_stream()
            body = asyncio.run(read_one_chunk())

        self.assertTrue(response.media_type.startswith("multipart/x-mixed-replace"))
        self.assertIn(b"--frame\r\n", body)
        self.assertIn(b"Content-Type: image/jpeg", body)
        self.assertIn(self.jpeg, body)

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

    def test_converts_valid_glb_to_downloadable_stl(self):
        source_mesh = trimesh.creation.box(extents=[0.04, 0.02, 0.01])
        glb_bytes = source_mesh.export(file_type="glb")

        response = self.client.post(
            "/api/v1/mesh/glb-to-stl",
            files={"model": ("concept.glb", glb_bytes, "model/gltf-binary")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "model/stl")
        self.assertEqual(response.headers["x-geometry-mode"], "single-image-ai-concept-not-repair")
        self.assertEqual(response.headers["x-output-units"], "mm")
        converted = trimesh.load(io.BytesIO(response.content), file_type="stl", force="mesh")
        self.assertEqual(converted.extents.tolist(), [40, 20, 10])

    def test_rejects_invalid_glb_conversion_inputs(self):
        wrong_type = self.client.post(
            "/api/v1/mesh/glb-to-stl",
            files={"model": ("concept.obj", b"mesh", "application/octet-stream")},
        )
        invalid_glb = self.client.post(
            "/api/v1/mesh/glb-to-stl",
            files={"model": ("concept.glb", b"not a GLB", "model/gltf-binary")},
        )

        self.assertEqual(wrong_type.status_code, 415)
        self.assertEqual(invalid_glb.status_code, 400)

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