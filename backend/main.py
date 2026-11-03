"""Local API for validating camera frames and exporting measured concept geometry."""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from backend.camera_stream import DEFAULT_STREAM_URL, decode_jpeg_frame, iter_camera_frames
from backend.mesh_export import create_stepped_cornice

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png"}
CAMERA_STREAM_URL = os.environ.get("ESP32_STREAM_URL", DEFAULT_STREAM_URL)
CAMERA_STATUS_LOCK = threading.Lock()
CAMERA_FRAME_TIMES: list[float] = []
CAMERA_STATUS = {
    "state": "idle",
    "width": None,
    "height": None,
    "fps": 0.0,
    "frames_received": 0,
    "last_frame_at": None,
    "error": None,
}

app = FastAPI(title="Robo Rebels Heritage API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "geometry": "dimensioned concept mesh; no AI inference"}


@app.get("/api/v1/camera/status")
def camera_status() -> dict[str, str | int | float | None]:
    with CAMERA_STATUS_LOCK:
        return {**CAMERA_STATUS, "source": CAMERA_STREAM_URL}


@app.get("/api/v1/camera/stream")
def camera_stream() -> StreamingResponse:
    """Proxy the configured ESP32 MJPEG stream to a browser as multipart JPEG."""
    with CAMERA_STATUS_LOCK:
        CAMERA_STATUS.update(state="connecting", error=None)

    def frames():
        try:
            for frame in iter_camera_frames(CAMERA_STREAM_URL):
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    continue
                frame_height, frame_width = frame.shape[:2]
                now = time.time()
                with CAMERA_STATUS_LOCK:
                    CAMERA_FRAME_TIMES.append(now)
                    while CAMERA_FRAME_TIMES and now - CAMERA_FRAME_TIMES[0] > 5:
                        CAMERA_FRAME_TIMES.pop(0)
                    frame_count = CAMERA_STATUS["frames_received"] + 1
                    elapsed = CAMERA_FRAME_TIMES[-1] - CAMERA_FRAME_TIMES[0] if len(CAMERA_FRAME_TIMES) > 1 else 0
                    CAMERA_STATUS.update(
                        state="streaming",
                        width=frame_width,
                        height=frame_height,
                        fps=round((len(CAMERA_FRAME_TIMES) - 1) / elapsed, 1) if elapsed > 0 else 0.0,
                        frames_received=frame_count,
                        last_frame_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                        error=None,
                    )
                jpeg = encoded.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
        except Exception as error:
            with CAMERA_STATUS_LOCK:
                CAMERA_STATUS.update(state="error", error=str(error))
        else:
            with CAMERA_STATUS_LOCK:
                CAMERA_STATUS.update(state="stopped")

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/concept-mesh")
async def export_concept_mesh(
    image: UploadFile = File(...),
    width_mm: float = Form(...),
    height_mm: float = Form(...),
    depth_mm: float = Form(...),
) -> Response:
    """Validate one camera image and export a stepped mesh from measured dimensions."""
    if image.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded bytes are not a valid JPEG or PNG image")
    image_height, image_width = decoded.shape[:2]

    try:
        mesh = create_stepped_cornice(width_mm, height_mm, depth_mm)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    stl_bytes = mesh.export(file_type="stl")
    return Response(
        content=stl_bytes,
        media_type="model/stl",
        headers={
            "Content-Disposition": 'attachment; filename="cornice-concept.stl"',
            "X-Image-Resolution": f"{image_width}x{image_height}",
            "X-Geometry-Mode": "dimensioned-parametric-placeholder",
        },
    )