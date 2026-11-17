"""Measurement sheet: what was found (left) and exactly what the STL files fill (right)."""

from __future__ import annotations

import cv2
import numpy as np


def _label(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def draw_sheet(img: np.ndarray, target: np.ndarray, shapes: list[dict], parts: list[dict],
               scale_text: str, max_width: int = 2400) -> np.ndarray:
    left = img.copy()
    over = left.copy()
    over[target > 0] = (0, 0, 255)
    left = cv2.addWeighted(over, 0.4, left, 0.6, 0)
    cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, cnts, -1, (0, 0, 255), 2)
    for s in shapes:
        cv2.polylines(left, [np.round(s["poly"]).astype(np.int32)], True, (0, 220, 0), 1)

    right = np.full_like(img, 25)                      # ONLY the filled areas, nothing around them
    right[target > 0] = (245, 245, 245)
    for k, p in enumerate(parts):
        x1, y1, _, _ = p["box_px"]
        text = f'#{k + 1} {p["width_mm"]}x{p["length_mm"]}x{p["height_mm"]} mm'
        _label(left, text, (x1, max(40, y1 - 8)))
        _label(right, text, (x1, max(40, y1 - 8)))
    for panel, title in ((left, f"FOUND   scale: {scale_text}"), (right, "STL FILLS ONLY THIS")):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(panel, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    sheet = np.hstack([left, right])
    if sheet.shape[1] > max_width:
        f = max_width / sheet.shape[1]
        sheet = cv2.resize(sheet, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return sheet
