import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh

from backend.whitefill import analyze

BLUE, WHITE = (170, 92, 30), (236, 239, 240)
YELLOW = (40, 215, 245)
GREY_TABLE = (150, 150, 150)


def chipped_cube_scene(colour=WHITE, background=BLUE):
    scene = np.full((400, 600, 3), background, np.uint8)
    cube = np.zeros(scene.shape[:2], np.uint8)
    cv2.rectangle(cube, (100, 100), (260, 260), 255, -1)
    chip = np.zeros_like(cube)
    cv2.fillPoly(chip, [np.array([[200, 100], [260, 100], [260, 160]])], 255)
    scene[(cube > 0) & (chip == 0)] = colour
    return cv2.GaussianBlur(scene, (0, 0), 1.0)


class FixedPhotoTests(unittest.TestCase):
    def test_fixed_photo_fills_missing_corner_and_keeps_the_rest(self):
        scene = chipped_cube_scene()
        with tempfile.TemporaryDirectory() as folder:
            result = analyze([scene], folder, marker_mm=0, mm_per_px=0.5)
            self.assertTrue(result["ok"])
            self.assertEqual(result["repaired_pieces"], 1)
            fixed = cv2.imread(str(Path(folder) / result["fixed"]))

        self.assertEqual(fixed.shape, scene.shape)
        chip_pixel = fixed[125, 245].astype(int)
        self.assertLess(np.abs(chip_pixel - np.array(WHITE)).max(), 25)
        self.assertGreater(np.abs(scene[125, 245].astype(int) - np.array(WHITE)).max(), 60)
        far_background = np.abs(fixed[350, 500].astype(int) - scene[350, 500].astype(int)).max()
        self.assertLessEqual(far_background, 2)  # JPEG rounding only

    def test_intact_piece_makes_no_fixed_photo(self):
        scene = np.full((400, 600, 3), BLUE, np.uint8)
        scene[100:260, 100:260] = WHITE
        scene = cv2.GaussianBlur(scene, (0, 0), 1.0)
        with tempfile.TemporaryDirectory() as folder:
            result = analyze([scene], folder, marker_mm=0, mm_per_px=0.5)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["fixed"])

    def test_colour_mode_restores_a_chipped_yellow_cube(self):
        scene = chipped_cube_scene(YELLOW, GREY_TABLE)
        with tempfile.TemporaryDirectory() as folder:
            ignored = analyze([scene], folder + "/white", marker_mm=0, mm_per_px=0.5, fill="complete")
            yellow = analyze([scene], folder + "/yellow", marker_mm=0, mm_per_px=0.5, fill="complete", colour="yellow")
            fixed = cv2.imread(str(Path(folder) / "yellow" / yellow["fixed"]))

        self.assertFalse(ignored["ok"])
        self.assertTrue(yellow["ok"])
        self.assertEqual(yellow["colour"], "yellow")
        full_cm2 = 160 * 160 * 0.25 / 100
        self.assertAlmostEqual(yellow["parts"][0]["area_cm2"], full_cm2, delta=0.05 * full_cm2)
        self.assertLess(np.abs(fixed[125, 245].astype(int) - np.array(YELLOW)).max(), 30)

    def test_unknown_colour_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze([chipped_cube_scene()], tempfile.gettempdir() + "/espai-bad-colour", marker_mm=0, colour="purple")

    def test_complete_mode_restores_the_whole_chipped_cube(self):
        scene = chipped_cube_scene()
        with tempfile.TemporaryDirectory() as folder:
            white = analyze([scene], folder + "/white", marker_mm=0, mm_per_px=0.5, fill="white")
            whole = analyze([scene], folder + "/whole", marker_mm=0, mm_per_px=0.5, fill="complete")
            mesh = trimesh.load(Path(folder) / "whole" / whole["parts"][0]["stl"], force="mesh")

        self.assertTrue(whole["ok"])
        self.assertEqual(len(whole["parts"]), 1)
        full_cm2 = 160 * 160 * 0.25 / 100          # 160 px cube at 0.5 mm/px
        self.assertAlmostEqual(whole["parts"][0]["area_cm2"], full_cm2, delta=0.04 * full_cm2)
        self.assertLess(white["parts"][0]["area_cm2"], whole["parts"][0]["area_cm2"] - 0.2 * 6.0)
        self.assertTrue(mesh.is_watertight)

    def test_complete_mode_guesses_unclear_shapes_instead_of_skipping_them(self):
        scene = np.full((400, 600, 3), BLUE, np.uint8)
        cv2.circle(scene, (300, 200), 70, WHITE, -1)
        notch = np.array([[300, 190], [380, 185], [380, 275]])
        cv2.fillPoly(scene, [notch], BLUE)
        scene = cv2.GaussianBlur(scene, (0, 0), 1.0)
        with tempfile.TemporaryDirectory() as folder:
            skipped = analyze([scene], folder + "/white", marker_mm=0, mm_per_px=0.5, fill="white")
            guessed = analyze([scene], folder + "/whole", marker_mm=0, mm_per_px=0.5, fill="complete")
            fixed = cv2.imread(str(Path(folder) / "whole" / guessed["fixed"])) if guessed.get("fixed") else None

        self.assertFalse(skipped["ok"])
        self.assertTrue(guessed["ok"])
        self.assertTrue(guessed["parts"][0]["guessed"])
        self.assertEqual(guessed["guessed_pieces"], 1)
        self.assertIn("best-guess", guessed["description"])
        self.assertEqual(guessed["repaired_pieces"], 1)
        self.assertIsNotNone(fixed)
        self.assertLess(np.abs(fixed[220, 320].astype(int) - np.array(WHITE)).max(), 25)


if __name__ == "__main__":
    unittest.main()
