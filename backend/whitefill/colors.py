"""Pure-white detection. Any colour cast (cream, yellow, pink, grey, blue, bluish white) is NOT white."""

from __future__ import annotations

import cv2
import numpy as np


def pure_white_mask(img: np.ndarray) -> np.ndarray:
    """255 where the pixel is pure white: neutral (no tint), bright, and not a shadowed grey."""
    sm = cv2.GaussianBlur(img, (5, 5), 0)
    mx = sm.max(axis=2).astype(np.float32)
    mn = sm.min(axis=2).astype(np.float32)
    b, g, r = [sm[..., i].astype(np.float32) for i in range(3)]
    neutral = (mx - mn) < 24
    bright = np.percentile(mx[neutral], 95) if neutral.any() else 255
    local = cv2.dilate(np.where(neutral, mx, 0).astype(np.float32), np.ones((61, 61), np.uint8))
    m = neutral & (mx > max(120.0, 0.68 * bright)) & (mx > 0.8 * local) & ((b - (g + r) / 2) < 15)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# OpenCV hue is 0-179. Neutral greys and very pale tints fail the saturation test, so the table and shadows are ignored.
COLOUR_HUES = {"red": [(0, 9), (175, 179)], "orange": [(10, 21)], "yellow": [(22, 34)],
               "green": [(35, 84)], "blue": [(85, 129)], "pink": [(140, 174)]}
PIECE_COLOURS = ("white", *COLOUR_HUES)


def piece_colour_mask(img: np.ndarray, colour: str = "white") -> np.ndarray:
    """255 where the pixel belongs to a piece of the chosen colour ("white" keeps the strict pure-white test)."""
    if colour == "white":
        return pure_white_mask(img)
    if colour not in COLOUR_HUES:
        raise ValueError(f"colour must be one of {PIECE_COLOURS}")
    h, s, v = cv2.split(cv2.cvtColor(cv2.medianBlur(cv2.GaussianBlur(img, (5, 5), 0), 5), cv2.COLOR_BGR2HSV))
    hue = np.zeros(h.shape, bool)
    for lo, hi in COLOUR_HUES[colour]:
        hue |= (h >= lo) & (h <= hi)
    m = (hue & (s >= 30) & (v >= 60)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
