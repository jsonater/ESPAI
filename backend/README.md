# Camera and concept-mesh API

This local backend connects to the car's binary-JPEG WebSocket, validates frames with OpenCV, re-serves them as browser MJPEG, and exports a stepped concept STL using Trimesh. The image is currently **not** analyzed to infer shape or dimensions. Width, height, and depth must be measured and supplied in millimeters. The output is a placeholder for workflow testing, not a reconstruction or a print-ready restoration part.

## Install and run

From the project directory on Linux Mint:

```sh
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open the local dashboard at `http://127.0.0.1:8000/`. The health endpoint is `http://127.0.0.1:8000/api/health`; interactive API docs are available at `/docs`.

At startup, the backend creates one WebSocket client task that reconnects to `ws://192.168.4.1/Camera` and caches each binary JPEG. The dashboard's **Live camera** view consumes the backend's `GET /api/v1/camera/stream` MJPEG proxy and polls `GET /api/v1/camera/status`.

```sh
ESP32_WEBSOCKET_URL="ws://192.168.4.1/Camera" \
ESP32_CAPTURE_URL="http://192.168.4.1/capture" \
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The **Capture still** button saves the newest WebSocket frame as a timestamped JPEG in `~/Desktop/ESP car photos/` and stages the same image in the local dataset label form. If no WebSocket frame has arrived, it tries the optional HTTP `/capture` endpoint. The folder is created on first capture and contains photos only. These camera operations work on the isolated car Wi-Fi without internet.

## Local image analysis with Ollama

The Live camera view can use a local Ollama vision model for image-level condition descriptions. Install Ollama 0.12.7 or later, then download the default model once:

```sh
ollama pull qwen3-vl:2b
```

The model download is about 1.9 GB; inference needs additional working memory. It should be a reasonable starting point on a 12 GB laptop, but close other memory-heavy applications if the system becomes slow. Keep the Ollama service running and start the backend as above. The default model can be changed with `OLLAMA_MODEL`; the local service URL can be changed with `OLLAMA_BASE_URL`.

The backend checks `GET /api/v1/ollama/status` and sends one selected frame to `POST /api/v1/ollama/predict`. You can also choose a JPEG/PNG with **Choose photo** and click **Analyze photo** while the car is away; no camera connection is needed for that workflow. With the default `http://127.0.0.1:11434` URL, image bytes are sent only to Ollama on this computer; if `OLLAMA_BASE_URL` is changed, they are sent to that configured service instead. This model can describe visible damage and classify its apparent condition; it cannot establish physical scale, infer hidden geometry, reconstruct a missing part, or generate a repaired STL. Treat results as suggestions for review, not restoration instructions. The separate Roboflow option remains cloud-based and requires internet.

Roboflow Serverless Cloud inference is one frame per explicit button press, not automatic per-frame polling, and requires internet. Create or train a model in Roboflow, then configure its model ID (`project/version`) and API key in the server process environment when internet is available:

```sh
ROBOFLOW_API_KEY="your-key" ROBOFLOW_MODEL_ID="your-project/1" \
ESP32_WEBSOCKET_URL="ws://192.168.4.1/Camera" \
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Type the key directly in your terminal; never put it in `index.html`, commit it, or send it in chat. The backend keeps the key server-side and uses Roboflow's Serverless Cloud API with the recommended Bearer-header transport. `GET /api/v1/roboflow/status` reports whether model settings are present; `POST /api/v1/roboflow/predict` accepts one JPEG/PNG image. Each request may use Roboflow inference credits.

When connected only to the car's access point, capture and label images locally; reconnect to internet Wi-Fi later to upload data or use Roboflow Cloud. Do not put school Wi-Fi credentials into firmware without the school's approval. This API binds to localhost and is intended for use from the same laptop.

The standalone `backend.camera_stream` utility remains for cameras that expose an HTTP MJPEG endpoint; it is not the path used by the `/Camera` WebSocket bridge:

```sh
ESP32_STREAM_URL="http://192.168.4.1/stream" python -m backend.camera_stream
```

The default IP is only an example; use the address and stream route printed by your ESP32 firmware. Save one decoded frame for inspection with `--save-first data/first-frame.jpg`. The frame iterator is the integration point for a future inference model.

The API binds to localhost by default. To let the ESP32 reach it over a trusted local network, use `--host 0.0.0.0` and allow port 8000 through the laptop firewall. This prototype has no authentication; do not expose it to the public internet.

## Tests

```sh
python -m unittest discover -s backend/tests -v
```