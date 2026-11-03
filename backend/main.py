"""Local API for validating camera frames and exporting measured concept geometry."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend.camera_stream import DEFAULT_STREAM_URL, decode_jpeg_frame, iter_camera_frames
from backend.mesh_export import create_stepped_cornice

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png"}
CAMERA_STREAM_URL = os.environ.get("ESP32_STREAM_URL", DEFAULT_STREAM_URL)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html", media_type="text/html")


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


@app.get("/api/v1/roboflow/status")
def roboflow_status() -> dict[str, str | bool | None]:
    configured = bool(ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID)
    return {
        "configured": configured,
        "model_id": ROBOFLOW_MODEL_ID if configured else None,
        "runtime": "serverless-cloud" if configured else None,
    }


def run_roboflow_inference(image_rgb: np.ndarray) -> dict:
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY,
    ).configure(InferenceConfiguration(api_key_transport="header"))
    return client.infer(image_rgb, model_id=ROBOFLOW_MODEL_ID)


@app.post("/api/v1/roboflow/predict")
async def predict_with_roboflow(image: UploadFile = File(...)) -> dict:
    if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
        raise HTTPException(
            status_code=503,
            detail="Roboflow is not configured; set ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID before starting the API",
        )
    if image.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit")

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded bytes are not a valid JPEG or PNG image")
    image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    try:
        result = await run_in_threadpool(run_roboflow_inference, image_rgb)
    except Exception as error:
        raise HTTPException(status_code=502, detail="Roboflow inference failed; check the model, key, and internet connection") from error
    return result


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