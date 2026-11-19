"""API routes for whitefill: photos of pure-white cubes/triangles -> printable STL files.

Kept separate from backend.main so it can also run alone on a laptop (see whitefill_laptop.py).
Each scan is written to its own folder so no folder ever holds more than 20 items.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend.whitefill import analyze
from backend.whitefill.colors import PIECE_COLOURS
from backend.whitefill.detect import FILL_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "whitefill"      # data/ is git-ignored
MARKER_FILE = PROJECT_ROOT / "markers" / "marker_50mm.png"
MAX_PHOTOS = 8
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_ITEMS_PER_FOLDER = 20
SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+(\.(stl|jpg))?$")

router = APIRouter()


def new_scan_folder(root: Path | None = None) -> Path:
    """data/whitefill/batch_0001/scan_<time>/ ; a new batch starts when one holds 20 scans."""
    root = root or OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    batches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    batch = batches[-1] if batches else root / "batch_0001"
    if batch.exists() and sum(1 for _ in batch.iterdir()) >= MAX_ITEMS_PER_FOLDER:
        batch = root / f"batch_{int(batch.name.split('_')[1]) + 1:04d}"
    batch.mkdir(exist_ok=True)
    number = sum(1 for _ in batch.iterdir()) + 1
    while True:                                           # never reuse a folder, even within one second
        scan = batch / f"scan_{time.strftime('%Y%m%d_%H%M%S')}_{number:02d}"
        try:
            scan.mkdir()
            return scan
        except FileExistsError:
            number += 1


@router.get("/whitefill")
def whitefill_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "whitefill.html", media_type="text/html")


@router.get("/api/v1/whitefill/marker")
def whitefill_marker() -> FileResponse:
    return FileResponse(MARKER_FILE, media_type="image/png", filename="marker_50mm.png")


@router.get("/api/v1/whitefill/scans")
def whitefill_scans() -> dict:
    scans = []
    if not OUTPUT_ROOT.is_dir():
        return {"scans": scans}
    for batch in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
        if batch.is_symlink() or not batch.is_dir() or not re.fullmatch(r"batch_\d+", batch.name):
            continue
        for scan in sorted(batch.iterdir(), reverse=True):
            if scan.is_symlink() or not scan.is_dir() or not re.fullmatch(r"scan_\d{8}_\d{6}_\d+", scan.name):
                continue
            files = [file.name for file in sorted(scan.iterdir())
                     if not file.is_symlink() and file.is_file() and SAFE_NAME.fullmatch(file.name)
                     and file.suffix in {".stl", ".jpg"}]
            if files:
                scans.append({"batch": batch.name, "scan": scan.name, "files": files})
    return {"scans": scans}


@router.post("/api/v1/whitefill/analyze")
async def whitefill_analyze(
    photos: list[UploadFile] = File(...),
    fill: str = Form("white"),
    colour: str = Form("white"),
    marker_mm: float = Form(50.0),
    mm_per_px: float = Form(0.5),
    clearance_mm: float = Form(-1.0),
    height_mm: float = Form(0.0),
    min_piece_mm: float = Form(5.0),
    max_piece_mm: float = Form(120.0),
) -> dict:
    if fill not in FILL_MODES:
        raise HTTPException(status_code=422, detail=f"fill must be one of {', '.join(FILL_MODES)}")
    if colour not in PIECE_COLOURS:
        raise HTTPException(status_code=422, detail=f"colour must be one of {', '.join(PIECE_COLOURS)}")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(status_code=413, detail=f"Send at most {MAX_PHOTOS} photos")
    if not (0 <= marker_mm <= 500 and 0 < mm_per_px <= 50 and 0 <= height_mm <= 500
            and 0 < min_piece_mm < max_piece_mm <= 2000 and clearance_mm <= 5):
        raise HTTPException(status_code=422, detail="A setting is out of range")
    images = []
    image_scales = []
    for photo in photos:
        data = await photo.read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(status_code=413, detail="Each photo must be 10 MiB or smaller")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(1.0, 1600 / max(image.shape[:2]))
        images.append(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image)
        image_scales.append(scale)
    if not images:
        raise HTTPException(status_code=400, detail="No readable JPEG/PNG photo was sent")

    scan = new_scan_folder()
    result = await run_in_threadpool(
        analyze, images, str(scan), marker_mm=marker_mm, mm_per_px=mm_per_px, fill=fill,
        clearance_mm=None if clearance_mm < 0 else clearance_mm, height_mm=height_mm,
        min_piece_mm=min_piece_mm, max_piece_mm=max_piece_mm, colour=colour, image_scales=image_scales)
    result["files_url"] = f"/api/v1/whitefill/files/{scan.parent.name}/{scan.name}"
    return result


@router.get("/api/v1/whitefill/files/{batch}/{scan}/{name}")
def whitefill_file(batch: str, scan: str, name: str, download: bool = False) -> FileResponse:
    if not all(SAFE_NAME.match(part) for part in (batch, scan, name)):
        raise HTTPException(status_code=404, detail="File not found")
    path = OUTPUT_ROOT / batch / scan / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media = "model/stl" if name.endswith(".stl") else "image/jpeg"
    return FileResponse(path, media_type=media, filename=name if download else None)
