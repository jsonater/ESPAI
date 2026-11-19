"""Per-photo detection: which pixels should the printed part fill?"""

from __future__ import annotations

import cv2
import numpy as np

from .pieces import find_pieces
from .shapes import classify_piece

FILL_MODES = ("white", "missing", "complete")


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


def _paint_missing(fixed: np.ndarray, img: np.ndarray, piece: dict, gap: np.ndarray,
                   complete: np.ndarray) -> tuple[np.ndarray, bool]:
    if not gap.any():
        return fixed, False
    piece_colour = np.median(img[piece["mask"] > 0], axis=0)
    paint = cv2.dilate(gap, _ellipse(2)) & complete
    alpha = cv2.GaussianBlur(paint.astype(np.float32) / 255.0, (0, 0), 1.0)[..., None]
    blended = np.rint(fixed * (1 - alpha) + piece_colour * alpha).astype(np.uint8)
    return np.where(alpha > 0.004, blended, fixed), True


def detect(img: np.ndarray, valid: np.ndarray | None, px_per_mm: float | None, fill: str = "white",
           min_mm: float = 5.0, max_mm: float = 120.0, colour: str = "white") -> dict:
    """Find pure-white cubes and triangles.

    fill="white":   the part fills exactly the pure-white area of each piece, nothing around it.
    fill="missing": the part is what is missing from each piece's full cube/triangle outline.
    fill="complete": the whole piece with its missing parts restored; unclear shapes get a best-guess
                     (convex) outline instead of being skipped.
    """
    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}")
    pieces, skipped = find_pieces(img, valid, px_per_mm, min_mm, max_mm, colour)
    target = np.zeros(img.shape[:2], np.uint8)
    shapes = []
    fixed = img.copy()
    repaired = 0
    guessed = 0
    for piece in pieces:
        cls = classify_piece(piece, img.shape[:2])
        if cls is None:
            if fill != "complete":
                skipped["other_shape"] += 1
                continue
            hull = cv2.convexHull(piece["cnt"])
            guess = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(guess, [hull], 255)
            if valid is not None:
                guess = cv2.bitwise_and(guess, valid)
            (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
            target |= guess
            guess_gap = cv2.bitwise_and(guess, cv2.bitwise_not(piece["mask"]))
            fixed, repaired_guess = _paint_missing(fixed, img, piece, guess_gap, guess)
            repaired += int(repaired_guess)
            guessed += 1
            shapes.append({"kind": "piece", "poly": hull.reshape(-1, 2).astype(float).tolist(),
                           "side_px": float(np.sqrt(cv2.contourArea(hull))), "sides_px": [float(rw), float(rh)],
                           "has_part": True, "guessed": True})
            continue
        kind, _, complete, poly, side_px, sides, _ = cls
        gap = _missing_part(piece, complete, side_px, valid)
        if fill == "white":
            part = piece["mask"]
        elif fill == "missing":
            part = gap
        else:
            part = cv2.bitwise_or(piece["mask"], gap)
        target |= part
        fixed, repaired_piece = _paint_missing(fixed, img, piece, gap, complete)
        repaired += int(repaired_piece)
        shapes.append({"kind": kind, "poly": np.asarray(poly, float).tolist(), "side_px": float(side_px),
                       "sides_px": [float(v) for v in sides], "has_part": bool(part.any())})
    return {"target": target, "shapes": shapes, "skipped": skipped, "fixed": fixed, "repaired": repaired,
            "guessed": guessed}
