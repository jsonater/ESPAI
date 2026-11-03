"""Read and decode JPEG frames from an ESP32-CAM MJPEG stream."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
DEFAULT_STREAM_URL = "http://192.168.4.1/stream"
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024


class CameraStreamError(Exception):
    """Raised when the camera stream cannot provide a valid JPEG frame."""


def decode_jpeg_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decode one JPEG payload as an OpenCV BGR frame."""
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise CameraStreamError("Camera stream contained an invalid JPEG frame")
    return frame


def iter_jpeg_frames(
    stream: BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> Iterator[bytes]:
    """Yield complete JPEGs from an MJPEG byte stream with bounded buffering."""
    if chunk_size < 1 or max_frame_bytes < 4:
        raise ValueError("chunk_size must be positive and max_frame_bytes must be at least 4")

    buffer = bytearray()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)

        while True:
            frame_start = buffer.find(JPEG_START)
            if frame_start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if frame_start:
                del buffer[:frame_start]

            frame_end = buffer.find(JPEG_END, len(JPEG_START))
            if frame_end < 0:
                if len(buffer) > max_frame_bytes:
                    raise CameraStreamError(f"JPEG frame exceeds {max_frame_bytes} bytes")
                break

            frame_length = frame_end + len(JPEG_END)
            if frame_length > max_frame_bytes:
                raise CameraStreamError(f"JPEG frame exceeds {max_frame_bytes} bytes")
            yield bytes(buffer[:frame_length])
            del buffer[:frame_length]

    if buffer.find(JPEG_START) >= 0:
        raise CameraStreamError("Camera stream ended in the middle of a JPEG frame")


def iter_camera_frames(
    url: str,
    *,
    timeout: float = 10.0,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> Iterator[np.ndarray]:
    """Connect to an ESP32 MJPEG URL and yield decoded BGR frames."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "multipart/x-mixed-replace,image/jpeg"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise CameraStreamError(f"Camera returned HTTP status {status}")

        for jpeg_bytes in iter_jpeg_frames(response, max_frame_bytes=max_frame_bytes):
            yield decode_jpeg_frame(jpeg_bytes)


def send_frame_to_mesh_api(
    frame: np.ndarray,
    api_url: str,
    *,
    width_mm: float,
    height_mm: float,
    depth_mm: float,
    output_path: Path,
    timeout: float = 30.0,
) -> None:
    """POST one JPEG frame and measured dimensions; save the returned STL."""
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise CameraStreamError("Could not encode camera frame as JPEG")

    boundary = f"----RoboRebels{secrets.token_hex(16)}"
    body_parts: list[bytes] = []
    fields = {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "depth_mm": depth_mm,
    }
    for field_name, field_value in fields.items():
        body_parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode(),
            f"{field_value}\r\n".encode(),
        ])
    body_parts.extend([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="esp32-frame.jpg"\r\n',
        b"Content-Type: image/jpeg\r\n\r\n",
        encoded_image.tobytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        api_url,
        data=b"".join(body_parts),
        headers={
            "Accept": "model/stl",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                raise CameraStreamError(f"Mesh API returned HTTP status {response.status}")
            content_type = response.headers.get_content_type()
            if content_type != "model/stl":
                detail = response.read(1024).decode("utf-8", errors="replace")
                raise CameraStreamError(f"Mesh API did not return an STL file: {detail}")
            stl_bytes = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(1024).decode("utf-8", errors="replace")
        raise CameraStreamError(f"Mesh API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise CameraStreamError(f"Could not reach mesh API: {error.reason}") from error

    if not stl_bytes:
        raise CameraStreamError("Mesh API returned an empty STL file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(stl_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("ESP32_STREAM_URL", DEFAULT_STREAM_URL),
        help="ESP32-CAM MJPEG URL (or set ESP32_STREAM_URL)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Socket timeout in seconds")
    parser.add_argument("--max-frame-mb", type=float, default=8.0, help="Maximum decoded JPEG size in MiB")
    parser.add_argument("--save-first", help="Save the first decoded frame to this image path")
    parser.add_argument("--export-url", help="POST the first frame to the concept-mesh API")
    parser.add_argument("--width-mm", type=float, help="Measured part width for STL generation")
    parser.add_argument("--height-mm", type=float, help="Measured part height for STL generation")
    parser.add_argument("--depth-mm", type=float, help="Measured part depth for STL generation")
    parser.add_argument("--output-stl", default="cornice-concept.stl", help="Path for the returned STL")
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_frame_mb <= 0:
        parser.error("timeout and max-frame-mb must be positive")
    dimensions = (args.width_mm, args.height_mm, args.depth_mm)
    if args.export_url and any(value is None for value in dimensions):
        parser.error("--export-url requires --width-mm, --height-mm, and --depth-mm")
    if not args.export_url and any(value is not None for value in dimensions):
        parser.error("dimension options require --export-url")
    return args


def main() -> int:
    args = parse_args()
    max_frame_bytes = int(args.max_frame_mb * 1024 * 1024)
    frame_count = 0
    try:
        for frame in iter_camera_frames(
            args.url,
            timeout=args.timeout,
            max_frame_bytes=max_frame_bytes,
        ):
            frame_count += 1
            if frame_count == 1:
                print(f"[INFO] Connected to ESP32-CAM stream: {args.url}")
            if args.save_first and frame_count == 1:
                if not cv2.imwrite(args.save_first, frame):
                    raise CameraStreamError(f"Could not save first frame to {args.save_first}")
                print(f"[INFO] Saved first frame to {args.save_first}")
            if args.export_url and frame_count == 1:
                send_frame_to_mesh_api(
                    frame,
                    args.export_url,
                    width_mm=args.width_mm,
                    height_mm=args.height_mm,
                    depth_mm=args.depth_mm,
                    output_path=Path(args.output_stl),
                    timeout=args.timeout,
                )
                print(f"[INFO] Saved concept STL to {args.output_stl}")
                break
            if frame_count == 1 or frame_count % 100 == 0:
                height, width = frame.shape[:2]
                print(f"[INFO] Received frame {frame_count}: {width} x {height}")
    except (urllib.error.URLError, TimeoutError, OSError, CameraStreamError, ValueError) as error:
        print(f"[ERROR] Could not read ESP32-CAM stream: {error}", file=sys.stderr)
        return 1

    if frame_count == 0:
        print("[ERROR] Stream ended without a complete JPEG frame", file=sys.stderr)
        return 1
    print(f"[INFO] Stream ended after {frame_count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
