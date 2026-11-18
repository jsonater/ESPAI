import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import backend.main as api
import backend.whitefill_api as whitefill_api


class WhitefillApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        ok, encoded = cv2.imencode(".jpg", np.full((80, 120, 3), 255, dtype=np.uint8))
        if not ok:
            raise RuntimeError("Could not encode JPEG test image")
        cls.jpeg = encoded.tobytes()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = patch.object(whitefill_api, "OUTPUT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_whitefill_page_and_marker_are_served(self):
        page = self.client.get("/whitefill")
        marker = self.client.get("/api/v1/whitefill/marker")

        self.assertEqual(page.status_code, 200)
        self.assertIn("White fill", page.text)
        self.assertEqual(marker.status_code, 200)
        self.assertEqual(marker.headers["content-type"], "image/png")

    def test_whitefill_analyze_rejects_invalid_requests(self):
        invalid_fill = self.client.post(
            "/api/v1/whitefill/analyze",
            files=[("photos", ("a.jpg", self.jpeg, "image/jpeg"))],
            data={"fill": "blue"},
        )
        too_many = self.client.post(
            "/api/v1/whitefill/analyze",
            files=[("photos", (f"{i}.jpg", self.jpeg, "image/jpeg")) for i in range(9)],
        )

        self.assertEqual(invalid_fill.status_code, 422)
        self.assertEqual(too_many.status_code, 413)

    def test_whitefill_analyze_returns_files_url(self):
        fake_result = {
            "ok": True,
            "fill": "white",
            "photos_total": 1,
            "photos_measured": 1,
            "marker_found": False,
            "scale_source": "manual",
            "skipped": {},
            "parts": [{"stl": "part_01.stl", "faces": 12}],
            "sheet": "sheet.jpg",
        }
        with patch("backend.whitefill_api.analyze", return_value=fake_result):
            response = self.client.post(
                "/api/v1/whitefill/analyze",
                files=[("photos", ("a.jpg", self.jpeg, "image/jpeg"))],
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertIn("/api/v1/whitefill/files/batch_", body["files_url"])
        self.assertIn("/scan_", body["files_url"])

    def test_whitefill_file_download_and_path_safety(self):
        scan = self.root / "batch_0001" / "scan_20261117_101010_01"
        scan.mkdir(parents=True, exist_ok=True)
        (scan / "part_01.stl").write_bytes(b"solid test")

        file_ok = self.client.get("/api/v1/whitefill/files/batch_0001/scan_20261117_101010_01/part_01.stl")
        file_bad = self.client.get("/api/v1/whitefill/files/batch_0001/scan_20261117_101010_01/..%2Fsecret.stl")

        self.assertEqual(file_ok.status_code, 200)
        self.assertEqual(file_ok.headers["content-type"], "model/stl")
        self.assertEqual(file_bad.status_code, 404)

    def test_scans_lists_only_saved_whitefill_results(self):
        scan = self.root / "batch_0001" / "scan_20261117_101010_01"
        scan.mkdir(parents=True)
        (scan / "part_01.stl").write_bytes(b"stl")
        (scan / "sheet.jpg").write_bytes(b"jpg")
        (scan / "private.txt").write_text("not a result")
        response = self.client.get("/api/v1/whitefill/scans")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"scans": [{"batch": "batch_0001", "scan": scan.name,
                                                        "files": ["part_01.stl", "sheet.jpg"]}]})

    def test_new_scan_folder_starts_new_batch_after_twenty(self):
        scans = [whitefill_api.new_scan_folder(self.root) for _ in range(45)]
        self.assertEqual(len({path for path in scans}), 45)
        for batch in self.root.iterdir():
            if batch.is_dir():
                self.assertLessEqual(len(list(batch.iterdir())), whitefill_api.MAX_ITEMS_PER_FOLDER)


if __name__ == "__main__":
    unittest.main()
