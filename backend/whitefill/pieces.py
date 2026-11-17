"""Find separate small pure-white pieces and trace their true (half-way) edges."""

from __future__ import annotations

import cv2
import numpy as np

from .colors import pure_white_mask


def _fill(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask); cv2.drawContours(out, cnts, -1, 255, -1)
    return out, cnts


def find_pieces(img, valid, px_per_mm=None, min_mm=5.0, max_mm=120.0):
    """Separate small, pure-coloured pieces. Skips: too big, too small, cut off by the photo edge,
    and 'mixed' areas where white is speckled or patterned with another colour."""
    W = pure_white_mask(img)
    vfull = valid if valid is not None else np.full(W.shape, 255, np.uint8)
    W = cv2.bitwise_and(W, vfull)
    border = cv2.bitwise_not(cv2.erode(vfull, np.ones((7, 7), np.uint8)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(W)
    pieces, skipped = [], {"big": 0, "tiny": 0, "mixed": 0, "cut_off": 0, "other_shape": 0}
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 40:
            continue
        size = max(w, h)
        if px_per_mm:
            if size / px_per_mm < min_mm:
                skipped["tiny"] += 1; continue
            if size / px_per_mm > max_mm:
                skipped["big"] += 1; continue
        elif a < 0.0003 * W.size:
            skipped["tiny"] += 1; continue
        elif a > 0.15 * W.size:
            skipped["big"] += 1; continue
        comp = (lab == i).astype(np.uint8) * 255
        if cv2.bitwise_and(cv2.dilate(comp, np.ones((5, 5), np.uint8)), border).any():
            skipped["cut_off"] += 1; continue
        filled, cnts = _fill(comp)
        fa = float((filled > 0).sum())
        holes = cv2.bitwise_and(filled, cv2.bitwise_not(comp))
        nh, _, hst, _ = cv2.connectedComponentsWithStats(holes)
        specks = [hst[j, cv2.CC_STAT_AREA] for j in range(1, nh) if 4 <= hst[j, cv2.CC_STAT_AREA] < 0.02 * fa]
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        jagged = perim ** 2 / (4 * np.pi * fa + 1e-6)
        if sum(specks) > 0.01 * fa or len(specks) >= 2 or jagged > 7:
            skipped["mixed"] += 1; continue                # white mixed with another colour -> ignored
        geo = _halfway_edge(img, comp)                       # true edge position (sub-pixel accurate)
        gf, gc = _fill(geo)
        pieces.append({"mask": geo, "pure": comp, "filled": gf, "cnt": max(gc, key=cv2.contourArea)})
    return pieces, skipped


def _halfway_edge(img, comp):
    """Pure white decides WHICH pixels are the piece; the real edge lies half-way between the piece's
    white and the background around it. Re-trace the piece at that half-way level."""
    x, y, w, h = cv2.boundingRect(comp); pad = 8
    H, W = comp.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    c = comp[y0:y1, x0:x1]
    lo = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0).min(axis=2).astype(np.float32)  # white high, colours low
    inner = cv2.erode(c, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(c, np.ones((11, 11), np.uint8)) > 0) & ~(cv2.dilate(c, np.ones((5, 5), np.uint8)) > 0)
    if inner.sum() < 10 or ring.sum() < 10:
        return comp
    wv, bv = np.median(lo[inner]), np.median(lo[ring])
    if wv - bv < 30:
        return comp
    near = cv2.dilate(c, np.ones((7, 7), np.uint8)) > 0
    half = ((lo > (wv + bv) / 2) & near).astype(np.uint8) * 255
    n, lab = cv2.connectedComponents(half)
    ids = np.unique(lab[(c > 0) & (half > 0)]); ids = ids[ids > 0]
    out = np.zeros_like(comp)
    out[y0:y1, x0:x1] = np.isin(lab, ids).astype(np.uint8) * 255
    return out if out.any() else comp
