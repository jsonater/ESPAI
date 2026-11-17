"""Printed ArUco marker: real-world scale and perspective correction."""

from __future__ import annotations

import cv2
import numpy as np


def detect_marker(img):
    """Sub-pixel corners of the largest ArUco marker, ordered TL,TR,BR,BL as seen upright."""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    c = max(corners, key=lambda c: cv2.contourArea(c.reshape(4, 2))).reshape(4, 2).astype(np.float32)
    edge_angle = lambda k: abs(np.arctan2(*(np.roll(c, -k, 0)[1] - np.roll(c, -k, 0)[0])[::-1]))
    return np.roll(c, -min(range(4), key=edge_angle), 0)     # marker may be taped rotated


def marker_frame(images, corners, marker_mm, max_side=2200):
    """Warp every photo that shows the marker into ONE flat, square-on frame in millimetres.
    Removes perspective (photos taken at an angle) and aligns all photos to each other."""
    m = float(marker_mm)
    dst = np.float32([[0, 0], [m, 0], [m, m], [0, m]])
    Hs, boxes, dens = {}, [], []
    for i, c in corners.items():
        H = cv2.getPerspectiveTransform(c, dst)
        h, w = images[i].shape[:2]
        pts = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
        boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        dens.append(np.mean([np.linalg.norm(c[j] - c[(j + 1) % 4]) for j in range(4)]) / m)
        Hs[i] = H
    x0, y0, x1, y1 = np.median(np.array(boxes), 0)
    lim = 12 * m                                           # ignore the far horizon of steep photos
    x0, y0, x1, y1 = max(x0, -lim), max(y0, -lim), min(x1, m + lim), min(y1, m + lim)
    ppmm = float(np.clip(np.median(dens), 1.0, 8.0))       # keep the camera's native resolution
    ppmm = min(ppmm, max_side / max(x1 - x0, y1 - y0))
    W, H = int((x1 - x0) * ppmm), int((y1 - y0) * ppmm)
    S = np.array([[ppmm, 0, -x0 * ppmm], [0, ppmm, -y0 * ppmm], [0, 0, 1]])
    out = {}
    for i, Hm in Hs.items():
        M = S @ Hm
        img = cv2.warpPerspective(images[i], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.full(images[i].shape[:2], 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        out[i] = (img, cv2.erode(valid, np.ones((9, 9), np.uint8)))
    return out, ppmm, (dst - [x0, y0]) * ppmm
