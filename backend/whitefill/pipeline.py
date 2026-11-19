"""Photos in -> one STL per pure-white cube/triangle out, sized in millimetres."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from .detect import detect
from .marker import detect_marker, marker_frame
from .mesh import prism_stl
from .sheet import draw_sheet

MAX_PARTS = 20          # never write more than 20 STL files for one scan


def _sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _components(mask: np.ndarray, min_px: int = 20) -> list[np.ndarray]:
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    ids = sorted((i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px),
                 key=lambda i: -st[i, cv2.CC_STAT_AREA])
    return [(lab == i).astype(np.uint8) * 255 for i in ids]


def _vote(targets: list[np.ndarray], valids: list[np.ndarray]) -> np.ndarray:
    """A pixel is kept when at least half of the photos that can see it agree."""
    if len(targets) == 1:
        return targets[0]
    num = sum((t > 0).astype(np.float32) for t in targets)
    den = sum((v > 0).astype(np.float32) for v in valids)
    return ((num / np.maximum(den, 1)) >= 0.5).astype(np.uint8) * 255


def _mad(values: list[float]) -> float:
    v = np.asarray(values, float)
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def analyze(images: list[np.ndarray], out_dir: str, *, marker_mm: float = 50.0, mm_per_px: float = 0.5,
            fill: str = "white", clearance_mm: float | None = None, height_mm: float = 0.0,
            min_piece_mm: float = 5.0, max_piece_mm: float = 120.0, colour: str = "white",
            image_scales: list[float] | None = None) -> dict:
    """Return a JSON-safe result dict; STL files and a measurement sheet are written to out_dir.

    height_mm 0 -> each part is as tall as its cube is wide (triangle: average side).
    clearance_mm None -> 0 for "white" and "complete" (exact), 0.4 for "missing" (so it slides in).
    """
    if not images:
        raise ValueError("No photos")
    if image_scales is not None and (len(image_scales) != len(images) or any(not 0 < scale <= 1 for scale in image_scales)):
        raise ValueError("Image scales must match the photos and be greater than zero and at most one")
    if clearance_mm is None:
        clearance_mm = 0.4 if fill == "missing" else 0.0
    os.makedirs(out_dir, exist_ok=True)

    # 1. one flat, square-on frame in millimetres (needs the printed marker)
    corners = {i: c for i, c in enumerate(detect_marker(im) for im in images) if c is not None} if marker_mm else {}
    origin = None
    if corners:
        frame, px_per_mm, mpoly = marker_frame(images, corners, marker_mm)
        views = [frame[i][0] for i in sorted(frame)]
        valids = [frame[i][1] for i in sorted(frame)]
        hide = np.zeros(views[0].shape[:2], np.uint8)        # the marker's white border is not a piece
        c = mpoly.mean(0)
        cv2.fillPoly(hide, [(c + (mpoly - c) * 1.5).astype(np.int32)], 255)
        valids = [cv2.bitwise_and(v, cv2.bitwise_not(hide)) for v in valids]
        scale_text, origin = f"marker {marker_mm:g} mm, perspective-corrected", mpoly[0]
    else:
        best_index = max(range(len(images)), key=lambda index: _sharpness(images[index]))
        best = images[best_index]
        views, valids = [best], [np.full(best.shape[:2], 255, np.uint8)]
        effective_mm_per_px = float(mm_per_px) / (image_scales[best_index] if image_scales is not None else 1.0)
        px_per_mm = 1.0 / effective_mm_per_px
        scale_text = f"manual {mm_per_px:g} mm/px on original photo (no marker seen: sizes are a guess)"
    mm = 1.0 / px_per_mm

    # 2. detect in every photo, 3. vote
    found = [detect(v, val, px_per_mm, fill, min_piece_mm, max_piece_mm, colour) for v, val in zip(views, valids)]
    target = _vote([f["target"] for f in found], valids)
    target = cv2.morphologyEx(target, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ref = max(range(len(views)), key=lambda i: (len(found[i]["shapes"]), _sharpness(views[i])))
    shapes, skipped = found[ref]["shapes"], found[ref]["skipped"]
    comps = _components(target)[:MAX_PARTS]
    ignored = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in skipped.items() if v) or "nothing"
    base = {"fill": fill, "colour": colour, "scale_source": scale_text, "mm_per_px": round(mm, 4), "marker_found": bool(corners),
            "photos_measured": len(views), "photos_total": len(images), "skipped": skipped,
            "shapes_checked": len(shapes), "clearance_mm": clearance_mm}
    if not comps:
        return {**base, "ok": False,
                "message": f"No {'pure-white' if colour == 'white' else colour} cube or triangle to fill was found (ignored: {ignored})."}

    # 4. one STL per piece
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts = []
    for k, comp in enumerate(comps):
        x, y, bw, bh = cv2.boundingRect(comp)
        m = cv2.moments(comp, binaryImage=True)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shape = next((s for s in shapes if cv2.pointPolygonTest(
            np.asarray(s["poly"], np.float32).reshape(-1, 1, 2), (float(cx), float(cy)), True) > -3), None)
        height = float(height_mm) if height_mm and height_mm > 0 else (shape["side_px"] * mm if shape else 10.0)
        crop = comp[y:y + bh, x:x + bw]
        cl_px = int(round(clearance_mm / mm))
        if cl_px > 0:
            shrunk = cv2.erode(cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT),
                               np.ones((2 * cl_px + 1,) * 2, np.uint8))[1:-1, 1:-1]
            crop = shrunk if shrunk.any() else crop
        cell = max(0.2, max(bw, bh) * mm / 250)
        cols, rows = max(2, round(bw * mm / cell)), max(2, round(bh * mm / cell))
        soft = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 0.8)
        inc = cv2.resize(soft, (cols, rows), interpolation=cv2.INTER_LINEAR if cols > bw else cv2.INTER_AREA) > 127
        if inc.sum() < 4:
            continue
        name = f"part_{stamp}_{len(parts) + 1}.stl"
        faces = prism_stl(inc, cell, height, os.path.join(out_dir, name))
        ys, xs = np.nonzero(inc)
        sizes = []
        for f in found:                                    # same piece as seen in each photo
            ov = cv2.bitwise_and(f["target"], cv2.dilate(comp, np.ones((9, 9), np.uint8)))
            if ov.any():
                _, _, a, b = cv2.boundingRect(ov)
                sizes.append((a * mm, b * mm))
        part = {"stl": name, "faces": int(faces), "kind": shape["kind"] if shape else "piece",
                "width_mm": round(float(xs.max() - xs.min() + 1) * cell, 1),
                "length_mm": round(float(ys.max() - ys.min() + 1) * cell, 1),
                "height_mm": round(height, 1),
                "area_cm2": round(float((comp > 0).sum()) * mm * mm / 100, 2),
                "volume_cm3": round(float(inc.sum()) * cell * cell * height / 1000, 2),
                "box_px": [int(x), int(y), int(x + bw), int(y + bh)],
                "spread_mm": round(max(_mad([a for a, _ in sizes]), _mad([b for _, b in sizes])), 1)
                if len(sizes) > 1 else None}
        if shape:
            part["shape_sides_mm"] = [round(v * mm, 1) for v in shape["sides_px"]]
            if shape.get("guessed"):
                part["guessed"] = True
        if origin is not None:
            part["pos_mm"] = [round(float(cx - origin[0]) * mm, 1), round(float(cy - origin[1]) * mm, 1)]
        parts.append(part)
    if not parts:
        return {**base, "ok": False, "message": "The white pieces are too small to print at this scale."}

    sheet_name = f"sheet_{stamp}.jpg"
    cv2.imwrite(os.path.join(out_dir, sheet_name), draw_sheet(views[ref], target, shapes, parts, scale_text))
    fixed_name = None
    if found[ref]["repaired"]:
        fixed_name = f"fixed_{stamp}.jpg"
        cv2.imwrite(os.path.join(out_dir, fixed_name), found[ref]["fixed"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cubes = sum(p["kind"] == "cube" for p in parts)
    triangles = sum(p["kind"] == "triangle" for p in parts)
    guessed = sum(bool(p.get("guessed")) for p in parts)
    what = {"white": "filled exactly", "missing": "with a missing part",
            "complete": "completed (missing parts restored)"}[fill]
    guess_note = f", {guessed} best-guess shape(s)" if guessed else ""
    return {**base, "ok": True, "parts": parts, "sheet": sheet_name, "fixed": fixed_name,
            "repaired_pieces": int(found[ref]["repaired"]), "guessed_pieces": guessed,
            "description": f"{cubes} {'pure-white' if colour == 'white' else colour} cube(s), {triangles} triangle(s){guess_note} {what}; "
                           f"{len(parts)} STL file(s) made; ignored: {ignored}."}
