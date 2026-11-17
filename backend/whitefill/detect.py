"""Per-photo detection: which pixels should the printed part fill?"""

from __future__ import annotations

import cv2
import numpy as np

from .pieces import find_pieces
from .shapes import classify_piece

FILL_MODES = ("white", "missing")


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)


def _missing_part(piece: dict, complete: np.ndarray, side_px: float, valid: np.ndarray | None) -> np.ndarray:
    """Full shape minus the white that is there. Thin edge slivers are noise, sharp tips are kept."""
    body = cv2.dilate(piece["mask"], _ellipse(1))
    gap = cv2.bitwise_and(complete, cv2.bitwise_not(body))
    if valid is not None:
        gap = cv2.bitwise_and(gap, valid)
    d = max(2.0, 0.035 * side_px)
    to_body = cv2.distanceTransform(cv2.bitwise_not(body), cv2.DIST_L2, 3)
    to_edge = cv2.distanceTransform(complete, cv2.DIST_L2, 3)
    raw = gap.copy()
    gap[(to_body <= d) & (to_edge <= d)] = 0
    gap = cv2.morphologyEx(gap, cv2.MORPH_OPEN, _ellipse(1))
    for _ in range(int(np.ceil(d))):
        gap = cv2.dilate(gap, _ellipse(1)) & raw
    gap = cv2.dilate(gap, _ellipse(1)) & complete & cv2.bitwise_not(piece["mask"])
    n, lab, st, _ = cv2.connectedComponentsWithStats(gap)
    keep = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] > max(20, 0.015 * (complete > 0).sum())]
    return np.isin(lab, keep).astype(np.uint8) * 255


def detect(img: np.ndarray, valid: np.ndarray | None, px_per_mm: float | None, fill: str = "white",
           min_mm: float = 5.0, max_mm: float = 120.0) -> dict:
    """Find pure-white cubes and triangles.

    fill="white":   the part fills exactly the pure-white area of each piece, nothing around it.
    fill="missing": the part is what is missing from each piece's full cube/triangle outline.
    """
    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}")
    pieces, skipped = find_pieces(img, valid, px_per_mm, min_mm, max_mm)
    target = np.zeros(img.shape[:2], np.uint8)
    shapes = []
    for piece in pieces:
        cls = classify_piece(piece, img.shape[:2])
        if cls is None:
            skipped["other_shape"] += 1
            continue
        kind, _, complete, poly, side_px, sides, _ = cls
        part = piece["mask"] if fill == "white" else _missing_part(piece, complete, side_px, valid)
        target |= part
        shapes.append({"kind": kind, "poly": np.asarray(poly, float).tolist(), "side_px": float(side_px),
                       "sides_px": [float(v) for v in sides], "has_part": bool(part.any())})
    return {"target": target, "shapes": shapes, "skipped": skipped}
