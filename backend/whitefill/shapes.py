"""Decide whether a piece is a cube (square/rectangle from above) or a triangle, and rebuild its full outline."""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

EDGE_PX = 0.0  # edges are already traced at the half-way level (see pieces._halfway_edge)


def _dominant_angle(cnt):
    """Main edge direction of a (roughly) square piece, from its straight edges."""
    ap = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(np.float64)
    seg = ap - np.roll(ap, 1, 0)
    L = np.hypot(seg[:, 0], seg[:, 1]); th = np.arctan2(seg[:, 1], seg[:, 0])
    return float(np.arctan2((L * np.sin(4 * th)).sum(), (L * np.cos(4 * th)).sum()) / 4)


def rect_completion(cnt, shape):
    """The full square/rectangle the piece should be (cubes), aligned with its own edges."""
    th = _dominant_angle(cnt)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = cnt.reshape(-1, 2).astype(np.float64); c0 = P.mean(0)
    Q = (P - c0) @ R                                         # rotate into the piece's own axes
    mn, mx = Q.min(0) - EDGE_PX, Q.max(0) + EDGE_PX
    box = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]]) @ R.T + c0
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(box).astype(np.int32)], 255)
    return m, box, (mx - mn), np.degrees(th)


def tri_completion(cnt, shape):
    """The full triangle: try combinations of the piece's straight edges (each re-fitted through all
    outline points along it), extend them until they meet, keep the tightest triangle that contains
    the whole piece."""
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    ap = cv2.approxPolyDP(hull, 0.015 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float64)
    if len(ap) < 3 or ha < 20:
        return None
    P = cnt.reshape(-1, 2).astype(np.float64); c0 = P.mean(0)
    Hp = hull.reshape(-1, 2).astype(np.float64)
    tol = max(2.0, 0.02 * np.sqrt(ha))
    cand = []
    for i in range(len(ap)):
        p, q = ap[i - 1], ap[i]
        L = np.hypot(*(q - p))
        if L < 0.12 * np.sqrt(ha):
            continue
        d = (q - p) / L; nrm = np.array([-d[1], d[0]])
        near = P[np.abs((P - p) @ nrm) < tol]
        cover = len(np.unique(np.floor((near - p) @ d / 2.0))) * 2.0 / L if len(near) else 0.0
        if cover >= 0.6 and len(near) >= 5:            # edge mostly intact: refit through all its points
            vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            p, d = np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)
        else:                                          # edge broken up (notch): use the longest REAL hull
            best_e = None                                  # edge in this direction (always contains the piece)
            for k in range(len(Hp)):
                a, b = Hp[k - 1], Hp[k]; le = np.hypot(*(b - a))
                if le < 3:
                    continue
                de = (b - a) / le
                if abs(de @ d) > np.cos(np.radians(8)) and (best_e is None or le > best_e[0]):
                    best_e = (le, a, de)
            if best_e is not None:
                p, d = best_e[1], best_e[2]
        n = np.array([-d[1], d[0]])
        if (c0 - p) @ n > 0:
            n = -n
        cand.append((p + n * EDGE_PX, d, np.degrees(np.arctan2(d[1], d[0])) % 180, L))
    cand = sorted(cand, key=lambda c: -c[3])[:8]

    def meet(l1, l2):
        A = np.array([l1[1], -l2[1]]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return l1[0] + np.linalg.solve(A, l2[0] - l1[0])[0] * l1[1]

    best = None
    Ps = P[::max(1, len(P) // 300)]
    for l1, l2, l3 in combinations(cand, 3):
        angs = [l1[2], l2[2], l3[2]]
        if any(min(abs(a1 - a2), 180 - abs(a1 - a2)) < 20 for a1, a2 in combinations(angs, 2)):
            continue
        V = [meet(l1, l2), meet(l2, l3), meet(l3, l1)]
        if any(v is None for v in V):
            continue
        tri = np.array(V, np.float32)
        area = abs(cv2.contourArea(tri))
        if not (0.9 * ha <= area <= 2.2 * ha):
            continue
        inside = np.mean([cv2.pointPolygonTest(tri.reshape(-1, 1, 2), (float(x), float(y)), True) >= -1.5
                          for x, y in Ps])
        if inside >= 0.97 and (best is None or area < best[0]):
            best = (area, tri.astype(np.float64))
    if best is None:
        return None
    tri = best[1]
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(tri).astype(np.int32)], 255)
    sides = [float(np.hypot(*(tri[i] - tri[i - 1]))) for i in range(3)]
    return m, tri, sides


def _on_edges(cnt, poly, tol):
    """Share of the piece's outline that lies ON the completed shape's straight edges."""
    P = cnt.reshape(-1, 2).astype(np.float32)
    poly = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    d = np.array([abs(cv2.pointPolygonTest(poly, (float(x), float(y)), True)) for x, y in P[::2]])
    return float((d <= tol).mean())


def classify_piece(p, shape):
    """'cube' (square/rectangle) or 'triangle' — anything else (circles, blobs) -> None."""
    fa = float((p["filled"] > 0).sum())
    cnt = p["cnt"]
    tol = max(2.0, 0.025 * np.sqrt(fa))
    best = None
    rm, rbox, rsides, ang = rect_completion(cnt, shape)
    rf = fa / max(1.0, float((rm > 0).sum()))
    ron = _on_edges(cnt, rbox, tol)
    if rf >= 0.7 and ron >= 0.55:
        best = ("cube", rf, rm, rbox, float(np.mean(rsides)), [float(rsides[0]), float(rsides[1])], ang)
    t = tri_completion(cnt, shape)
    if t is not None:
        tm, tri, tsides = t
        inside = float((cv2.bitwise_and(p["filled"], tm) > 0).sum()) / fa
        tf = fa / max(1.0, float((tm > 0).sum()))
        ton = _on_edges(cnt, tri, tol)
        if inside > 0.9 and tf >= 0.7 and ton >= 0.55 and (best is None or tf > best[1] + 0.05):
            best = ("triangle", tf, tm, tri, float(np.mean(tsides)), tsides, 0.0)
    return best
