import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from backend.camera_stream import (
    CameraStreamError,
    decode_jpeg_frame,
    iter_jpeg_frames,
    send_frame_to_mesh_api,
)


class JpegStreamTests(unittest.TestCase):
    def test_extracts_jpegs_when_markers_cross_chunk_boundaries(self):
        frames = [b"\xff\xd8first-frame\xff\xd9", b"\xff\xd8second-frame\xff\xd9"]

        decoded = list(iter_jpeg_frames(io.BytesIO(b"noise" + b"".join(frames)), chunk_size=3))

        self.assertEqual(decoded, frames)

    def test_rejects_a_frame_that_exceeds_the_limit(self):
        stream = io.BytesIO(b"\xff\xd8" + b"x" * 20)

        with self.assertRaisesRegex(CameraStreamError, "exceeds"):
            list(iter_jpeg_frames(stream, chunk_size=4, max_frame_bytes=8))

    def test_rejects_a_truncated_frame(self):
        with self.assertRaisesRegex(CameraStreamError, "middle of a JPEG"):
            list(iter_jpeg_frames(io.BytesIO(b"\xff\xd8partial")))

    def test_decodes_a_valid_jpeg_to_a_color_frame(self):
        success, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
        self.assertTrue(success)

        frame = decode_jpeg_frame(encoded.tobytes())

        self.assertEqual(frame.shape, (24, 32, 3))

    def test_rejects_invalid_jpeg_bytes(self):
        with self.assertRaisesRegex(CameraStreamError, "invalid JPEG"):
            decode_jpeg_frame(b"not a JPEG")

    def test_posts_first_frame_with_dimensions_and_saves_stl_response(self):
        success, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
        self.assertTrue(success)
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers.get_content_type.return_value = "model/stl"
        response.read.return_value = b"test-stl-data"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "concept.stl"
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                send_frame_to_mesh_api(
                    np.zeros((24, 32, 3), dtype=np.uint8),
                    "http://127.0.0.1:8000/api/v1/concept-mesh",
                    width_mm=90,
                    height_mm=42,
                    depth_mm=36,
                    output_path=output_path,
                )

            self.assertEqual(output_path.read_bytes(), b"test-stl-data")
            request = urlopen.call_args.args[0]
            self.assertIn(b'name="width_mm"', request.data)
            self.assertIn(b"90\r\n", request.data)
            self.assertIn(encoded.tobytes(), request.data)


if __name__ == "__main__":
    unittest.main()