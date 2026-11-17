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
