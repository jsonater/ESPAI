import base64
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_classifier import CONDITIONS, Sample, load_samples, stratified_indices, validate_class_counts


class DatasetBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bundle_path = Path(self.temp_dir.name) / "dataset.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_bundle(self, bundle):
        self.bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    def test_loads_supported_labels_and_skips_unknown(self):
        image_url = "data:image/png;base64," + base64.b64encode(b"png-data").decode("ascii")
        self.write_bundle({
            "format": "vestige-dataset-v1",
            "samples": [
                {"fileName": "wall.png", "condition": "Fracture", "site": "Site A", "imageDataUrl": image_url},
                {"fileName": "unknown.png", "condition": "Unknown", "imageDataUrl": image_url},
            ],
        })

        samples = load_samples(self.bundle_path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].condition, "Fracture")
        self.assertEqual(samples[0].image_bytes, b"png-data")

    def test_rejects_unsupported_image_format(self):
        self.write_bundle({
            "format": "vestige-dataset-v1",
            "samples": [{
                "fileName": "capture.heic",
                "condition": "Fracture",
                "imageDataUrl": "data:image/heic;base64,aGVsbG8=",
            }],
        })

        with self.assertRaisesRegex(ValueError, "JPEG or PNG"):
            load_samples(self.bundle_path)

    def test_requires_multiple_classes_and_minimum_examples(self):
        one_class = [Sample(b"x", "Fracture", "Site A", str(index)) for index in range(12)]
        with self.assertRaisesRegex(ValueError, "At least two"):
            validate_class_counts(one_class, 10)

        two_classes = [
            Sample(b"x", condition, "Site A", f"{condition}-{index}")
            for condition, count in (("Fracture", 4), ("Missing section", 12))
            for index in range(count)
        ]
        with self.assertRaisesRegex(ValueError, "at least 10 images"):
            validate_class_counts(two_classes, 10)

    def test_split_keeps_every_sample_and_each_class_in_all_splits(self):
        samples = [
            Sample(b"x", condition, "Site A", f"{condition}-{index}")
            for condition in CONDITIONS[:2]
            for index in range(20)
        ]

        splits = stratified_indices(samples, list(CONDITIONS[:2]), seed=7)

        flattened = [index for split in splits for index in split]
        self.assertEqual(sorted(flattened), list(range(len(samples))))
        for split in splits:
            self.assertEqual(Counter(samples[index].condition for index in split).keys(), set(CONDITIONS[:2]))


if __name__ == "__main__":
    unittest.main()