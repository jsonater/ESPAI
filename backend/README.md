# Camera and concept-mesh API

This local backend receives JPEG/PNG frames, validates them with OpenCV, and exports a stepped concept STL using Trimesh. The image is currently **not** analyzed to infer shape or dimensions. Width, height, and depth must be measured and supplied in millimeters. The output is a placeholder for workflow testing, not a reconstruction or a print-ready restoration part.

## Install and run

From the project directory on Linux Mint:

```sh
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open the local dashboard at `http://127.0.0.1:8000/`. The health endpoint is `http://127.0.0.1:8000/api/health`; interactive API docs are available at `/docs`.

The dashboard's **Live camera** view proxies `GET /api/v1/camera/stream` and polls `GET /api/v1/camera/status`. Configure the ESP32 stream URL before starting Uvicorn; the default `http://192.168.4.1/stream` is only an example:

```sh
ESP32_STREAM_URL="http://192.168.4.1/stream" uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Roboflow inference is one frame per explicit button press, not automatic per-frame polling. Create or train a model in Roboflow, then configure its model ID (`project/version`) and API key in the server process environment:

```sh
ROBOFLOW_API_KEY="your-key" ROBOFLOW_MODEL_ID="your-project/1" \
ESP32_STREAM_URL="http://192.168.4.1/stream" \
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Type the key directly in your terminal; never put it in `index.html`, commit it, or send it in chat. The backend keeps the key server-side and uses Roboflow's Serverless Cloud API with the recommended Bearer-header transport. `GET /api/v1/roboflow/status` reports whether model settings are present; `POST /api/v1/roboflow/predict` accepts one JPEG/PNG image. Each request may use Roboflow inference credits.

For cloud inference, connect the laptop and ESP32-CAM to the same Wi-Fi router or internet-enabled hotspot. If the laptop connects only to the ESP32's own access point, the local camera may work but Roboflow cloud inference may not have internet access. This API binds to localhost and is intended for use from the same laptop.

To read frames from an ESP32-CAM MJPEG server, set its actual stream URL and run the bridge from the project directory:

```sh
ESP32_STREAM_URL="http://192.168.4.1/stream" python -m backend.camera_stream
```

The default IP is only an example; use the address and stream route printed by your ESP32 firmware. Save one decoded frame for inspection with `--save-first data/first-frame.jpg`. The frame iterator is the integration point for a future inference model.

The API binds to localhost by default. To let the ESP32 reach it over a trusted local network, use `--host 0.0.0.0` and allow port 8000 through the laptop firewall. This prototype has no authentication; do not expose it to the public internet.

## Tests

```sh
python -m unittest discover -s backend/tests -v
```