"""Local API for validating camera frames and exporting measured concept geometry."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import struct
import threading
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import trimesh
import websockets
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend.camera_stream import CameraStreamError, decode_jpeg_frame
from backend.mesh_export import create_stepped_cornice

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_MODEL_BYTES = 50 * 1024 * 1024
CAMERA_FRAME_MAX_AGE_SECONDS = 2.0
CAMERA_RECEIVE_TIMEOUT_SECONDS = 5.0
IMAGE_TYPES = {"image/jpeg", "image/png"}
CAMERA_WEBSOCKET_URL = os.environ.get("ESP32_WEBSOCKET_URL", "ws://192.168.4.1/Camera")
CAMERA_CAPTURE_URL = os.environ.get("ESP32_CAPTURE_URL", "http://192.168.4.1/capture")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b")
OLLAMA_TIMEOUT_SECONDS = 180
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PHOTO_DIR = Path.home() / "Desktop" / "ESP car photos"
CAMERA_STATUS_LOCK = threading.Lock()
CAMERA_FRAME_TIMES: list[float] = []
CAMERA_LATEST_JPEG: bytes | None = None
CAMERA_FRAME_NUMBER = 0
CAMERA_LAST_FRAME_MONOTONIC: float | None = None
CAMERA_STATUS = {
    "state": "idle",
    "width": None,
    "height": None,
    "fps": 0.0,
    "frames_received": 0,
    "last_frame_at": None,
    "error": None,
}
OLLAMA_CONDITIONS = (
    "Missing section",
    "Fracture",
    "Surface erosion",
    "Intact reference",
    "Uncertain",
)


class OllamaUnavailableError(Exception):
    """Raised when the local Ollama service cannot be reached."""


class OllamaInferenceError(Exception):
    """Raised when Ollama cannot produce a valid structured result."""


class OllamaTimeoutError(Exception):
    """Raised when local vision inference exceeds the configured request timeout."""


async def receive_from_esp32() -> None:
    """Keep one WebSocket connection open and cache incoming binary JPEG frames."""
    global CAMERA_FRAME_NUMBER, CAMERA_LATEST_JPEG, CAMERA_LAST_FRAME_MONOTONIC
    reconnect_delay = 1
    while True:
        with CAMERA_STATUS_LOCK:
            state = "reconnecting" if CAMERA_STATUS["error"] else "connecting"
            CAMERA_STATUS.update(state=state)
        try:
            async with websockets.connect(
                CAMERA_WEBSOCKET_URL,
                open_timeout=5,
                close_timeout=3,
                ping_interval=20,
                max_size=MAX_UPLOAD_BYTES,
                compression=None,
                proxy=None,
            ) as websocket:
                reconnect_delay = 1
                with CAMERA_STATUS_LOCK:
                    CAMERA_FRAME_TIMES.clear()
                messages = aiter(websocket)
                while True:
                    try:
                        message = await asyncio.wait_for(anext(messages), timeout=CAMERA_RECEIVE_TIMEOUT_SECONDS)
                    except StopAsyncIteration:
                        raise CameraStreamError("Camera WebSocket closed")
                    except TimeoutError:
                        raise CameraStreamError("No camera frames received; reconnecting to the car")
                    if isinstance(message, str):
                        raise CameraStreamError("Expected binary JPEG WebSocket frames, received text")

                    jpeg = bytes(message)
                    frame = decode_jpeg_frame(jpeg)
                    height, width = frame.shape[:2]
                    now = time.time()
                    monotonic_now = time.monotonic()
                    with CAMERA_STATUS_LOCK:
                        CAMERA_FRAME_TIMES.append(now)
                        while CAMERA_FRAME_TIMES and now - CAMERA_FRAME_TIMES[0] > 5:
                            CAMERA_FRAME_TIMES.pop(0)
                        CAMERA_FRAME_NUMBER += 1
                        elapsed = CAMERA_FRAME_TIMES[-1] - CAMERA_FRAME_TIMES[0] if len(CAMERA_FRAME_TIMES) > 1 else 0
                        CAMERA_LATEST_JPEG = jpeg
                        CAMERA_LAST_FRAME_MONOTONIC = monotonic_now
                        CAMERA_STATUS.update(
                            state="streaming",
                            width=width,
                            height=height,
                            fps=round((len(CAMERA_FRAME_TIMES) - 1) / elapsed, 1) if elapsed > 0 else 0.0,
                            frames_received=CAMERA_FRAME_NUMBER,
                            last_frame_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                            error=None,
                        )
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            detail = str(error) or "WebSocket connection timed out"
            detail += ". Verify laptop Wi-Fi is MyWiFiCar, the car is powered on, and /Camera is the firmware WebSocket path"
            with CAMERA_STATUS_LOCK:
                CAMERA_STATUS.update(state="reconnecting", error=detail)
        except websockets.exceptions.InvalidStatus as error:
            response = error.response
            server = response.headers.get("Server")
            detail = f"WebSocket handshake rejected with HTTP {response.status_code}"
            if server:
                detail += f" by {server}"
            with CAMERA_STATUS_LOCK:
                CAMERA_STATUS.update(state="reconnecting", error=detail)
        except Exception as error:
            with CAMERA_STATUS_LOCK:
                CAMERA_STATUS.update(state="reconnecting", error=str(error))
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 8)


@asynccontextmanager
async def lifespan(app: FastAPI):
    receiver_task = asyncio.create_task(receive_from_esp32(), name="esp32-websocket-receiver")
    try:
        yield
    finally:
        receiver_task.cancel()
        with suppress(asyncio.CancelledError):
            await receiver_task


app = FastAPI(title="Robo Rebels Heritage API", version="0.1.0", lifespan=lifespan)
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
    return {"status": "ok", "geometry": "dimensioned concept mesh; image analysis does not reconstruct geometry"}


@app.get("/api/v1/camera/status")
def camera_status() -> dict[str, str | int | float | None]:
    with CAMERA_STATUS_LOCK:
        status = {**CAMERA_STATUS, "source": CAMERA_WEBSOCKET_URL, "capture_source": CAMERA_CAPTURE_URL}
        if status["state"] == "streaming" and not camera_frame_is_fresh():
            status.update(state="reconnecting", fps=0.0, error="No recent camera frames; waiting for the car")
        return status


def camera_frame_is_fresh() -> bool:
    return (
        CAMERA_STATUS["state"] == "streaming"
        and CAMERA_LATEST_JPEG is not None
        and CAMERA_LAST_FRAME_MONOTONIC is not None
        and time.monotonic() - CAMERA_LAST_FRAME_MONOTONIC <= CAMERA_FRAME_MAX_AGE_SECONDS
    )


def save_capture_photo(jpeg: bytes) -> Path:
    """Save one captured JPEG into the user's dedicated desktop photo folder."""
    CAPTURE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = CAPTURE_PHOTO_DIR / f"car-{timestamp}.jpg"
    path.write_bytes(jpeg)
    return path


@app.get("/api/v1/camera/frame")
def latest_camera_frame() -> Response:
    """Return a recent frame only while the camera WebSocket is streaming."""
    with CAMERA_STATUS_LOCK:
        jpeg = CAMERA_LATEST_JPEG if camera_frame_is_fresh() else None
        frame_number = CAMERA_FRAME_NUMBER
    if jpeg is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Number": str(frame_number),
        },
    )


@app.get("/api/v1/camera/capture")
def capture_camera_image() -> Response:
    """Return the latest WebSocket frame, or fetch one HTTP still as a fallback."""
    with CAMERA_STATUS_LOCK:
        latest_jpeg = CAMERA_LATEST_JPEG if camera_frame_is_fresh() else None
        latest_width = CAMERA_STATUS["width"]
        latest_height = CAMERA_STATUS["height"]
    if latest_jpeg is not None:
        try:
            saved_path = save_capture_photo(latest_jpeg)
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Could not save photo to Desktop/ESP car photos: {error}") from error
        return Response(
            content=latest_jpeg,
            media_type="image/jpeg",
            headers={
                "X-Image-Resolution": f"{latest_width}x{latest_height}",
                "X-Saved-Photo": saved_path.name,
            },
        )

    try:
        with urlopen(CAMERA_CAPTURE_URL, timeout=5) as response:
            image_bytes = response.read(MAX_UPLOAD_BYTES + 1)
    except (URLError, TimeoutError, OSError) as error:
        raise HTTPException(status_code=502, detail=f"Could not receive a camera still: {error}") from error

    if not image_bytes:
        raise HTTPException(status_code=502, detail="Camera capture endpoint returned an empty image")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Camera still exceeds the 10 MiB limit")

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=502, detail="Camera capture endpoint did not return a valid image")
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode the captured camera image")
    height, width = frame.shape[:2]
    jpeg = encoded.tobytes()
    try:
        saved_path = save_capture_photo(jpeg)
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not save photo to Desktop/ESP car photos: {error}") from error
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "X-Image-Resolution": f"{width}x{height}",
            "X-Saved-Photo": saved_path.name,
        },
    )


@app.get("/api/v1/camera/stream")
def camera_stream() -> StreamingResponse:
    """Fan out cached WebSocket JPEG frames to browser clients as MJPEG."""

    async def frames():
        last_frame_number = 0
        while True:
            with CAMERA_STATUS_LOCK:
                jpeg = CAMERA_LATEST_JPEG if camera_frame_is_fresh() else None
                frame_number = CAMERA_FRAME_NUMBER
            if jpeg is not None and frame_number != last_frame_number:
                last_frame_number = frame_number
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.02)

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
        "internet_required": True,
    }


@app.get("/api/v1/ollama/status")
def ollama_status() -> dict[str, str | bool | None]:
    request = Request(f"{OLLAMA_BASE_URL}/api/tags", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read(1024 * 1024 + 1))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {
            "available": False,
            "model_installed": False,
            "model": OLLAMA_MODEL,
            "runtime": "local",
            "message": "Start Ollama to enable local image analysis.",
        }

    models = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(models, list):
        models = []
    installed = any(
        isinstance(model, dict)
        and OLLAMA_MODEL in {model.get("name"), model.get("model")}
        for model in models
    )
    return {
        "available": True,
        "model_installed": installed,
        "model": OLLAMA_MODEL,
        "runtime": "local",
        "message": "Ready" if installed else f"Install the model with: ollama pull {OLLAMA_MODEL}",
    }


def run_ollama_inference(image_bytes: bytes) -> dict:
    request_body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": "In one short sentence, describe only what is visibly damaged in this image. If no damage is clear, say uncertain.",
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": 1024},
    }).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            response_body = response.read(1024 * 1024 + 1)
    except TimeoutError as error:
        raise OllamaTimeoutError(
            f"Local vision analysis exceeded {OLLAMA_TIMEOUT_SECONDS} seconds; try a smaller image or a faster model"
        ) from error
    except HTTPError as error:
        detail = error.read(1024).decode("utf-8", errors="replace")
        raise OllamaInferenceError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except (URLError, OSError) as error:
        raise OllamaUnavailableError("Could not reach local Ollama at 127.0.0.1:11434") from error

    if len(response_body) > 1024 * 1024:
        raise OllamaInferenceError("Ollama response exceeded the 1 MiB limit")
    try:
        payload = json.loads(response_body)
        observation = payload.get("response", "").strip()
        done_reason = payload.get("done_reason")
    except (json.JSONDecodeError, TypeError) as error:
        raise OllamaInferenceError("Ollama returned an invalid response") from error
    if not observation:
        detail = "Ollama returned no visible-damage description"
        if done_reason == "length":
            detail += " before reaching its output limit"
        raise OllamaInferenceError(detail)

    condition = next(
        (label for label in OLLAMA_CONDITIONS if observation.casefold().startswith(label.casefold())),
        "Uncertain",
    )
    analysis = {
        "condition": condition,
        "observations": observation,
        "visible_damage_location": "Not reliably determined by this image-level check.",
        "limitations": "Visual observations only; no physical scale, hidden geometry, reconstruction, or repaired STL.",
    }
    return {"provider": "ollama", "model": OLLAMA_MODEL, "analysis": analysis}


@app.post("/api/v1/ollama/predict")
async def predict_with_ollama(image: UploadFile = File(...)) -> dict:
    if not image.content_type or image.content_type.split(";")[0].strip().lower() not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit")
    if cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
        raise HTTPException(status_code=400, detail="Uploaded bytes are not a valid JPEG or PNG image")

    try:
        return await run_in_threadpool(run_ollama_inference, image_bytes)
    except OllamaTimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=f"{error}. Start Ollama and pull {OLLAMA_MODEL}.") from error
    except OllamaInferenceError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


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


@app.post("/api/v1/mesh/glb-to-stl")
async def convert_glb_to_stl(model: UploadFile = File(...)) -> Response:
    """Convert a generated GLB concept mesh to STL; this does not repair geometry."""
    filename = Path(model.filename or "").name
    if Path(filename).suffix.lower() != ".glb":
        raise HTTPException(status_code=415, detail="Upload a binary GLB model")
    if model.content_type not in {"model/gltf-binary", "application/octet-stream", ""}:
        raise HTTPException(status_code=415, detail="Upload a binary GLB model")

    model_bytes = await model.read(MAX_MODEL_BYTES + 1)
    await model.close()
    if not model_bytes:
        raise HTTPException(status_code=400, detail="Uploaded GLB is empty")
    if len(model_bytes) > MAX_MODEL_BYTES:
        raise HTTPException(status_code=413, detail="GLB exceeds the 50 MiB upload limit")

    try:
        if len(model_bytes) < 20 or model_bytes[:4] != b"glTF":
            raise ValueError("File does not have a GLB header")
        version, declared_length = struct.unpack_from("<II", model_bytes, 4)
        json_length, json_chunk_type = struct.unpack_from("<II", model_bytes, 12)
        if version != 2 or declared_length != len(model_bytes) or json_chunk_type != 0x4E4F534A:
            raise ValueError("GLB header or JSON chunk is invalid")
        if json_length > len(model_bytes) - 20:
            raise ValueError("GLB JSON chunk is truncated")
        glb_document = json.loads(model_bytes[20:20 + json_length].decode("utf-8").rstrip(" \x00"))
        resources = glb_document.get("buffers", []) + glb_document.get("images", [])
        if any(isinstance(resource, dict) and resource.get("uri") for resource in resources):
            raise ValueError("GLB references external files; upload a self-contained binary GLB")

        scene = trimesh.load(io.BytesIO(model_bytes), file_type="glb", force="scene")
        mesh = scene.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError("GLB contains no triangle mesh")
        if mesh.is_empty or not np.isfinite(mesh.vertices).all():
            raise ValueError("GLB contains no finite mesh geometry")
        mesh.apply_scale(1000.0)
        stl_bytes = mesh.export(file_type="stl")
        verified_mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
        if verified_mesh.is_empty or len(verified_mesh.faces) == 0:
            raise ValueError("Converted STL contains no triangles")
    except (ValueError, TypeError, KeyError, IndexError, UnicodeDecodeError, struct.error) as error:
        raise HTTPException(status_code=400, detail=f"Could not convert GLB mesh: {error}") from error

    return Response(
        content=stl_bytes,
        media_type="model/stl",
        headers={
            "Content-Disposition": 'attachment; filename="image-concept.stl"',
            "X-Mesh-Vertices": str(len(verified_mesh.vertices)),
            "X-Mesh-Faces": str(len(verified_mesh.faces)),
            "X-Mesh-Watertight": str(verified_mesh.is_watertight).lower(),
            "X-Output-Units": "mm",
            "X-Geometry-Mode": "single-image-ai-concept-not-repair",
        },
    )