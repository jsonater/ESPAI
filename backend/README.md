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

## Generate a 3D concept

The **Reconstruction** view embeds Stability AI's hosted [Stable Fast 3D](https://stabilityai-stable-fast-3d.hf.space/) generator in the app, so you do not need to switch to a separate app tab for the normal flow. This is separate from Ollama and requires internet. Upload a JPEG/PNG inside the embedded generator, choose **Remove Background** when offered, then choose **Run** and download the GLB result. In the same Reconstruction view, choose that `.glb` file, select **Convert to STL**, and then click **Download STL**. The conversion runs locally through `POST /api/v1/mesh/glb-to-stl`; uploads are limited to 50 MiB. If the hosted frame stalls, use **Open photo generator** and return to the same Reconstruction view to convert the downloaded GLB.

The photo is sent to Hugging Face/Stability AI only when you choose and upload it on their hosted page. The current Stability AI Community License permits research and non-commercial use; commercial use has additional terms. The app displays the required Stable Fast 3D attribution. A single-view model is a plausible 3D concept; hidden geometry, restoration, and generated color are model estimates, not verified facts. Keep the original GLB to retain any embedded color/materials. STL stores geometry only and does not preserve color. The converter scales GLB meters to STL millimeters, but the model's actual scale is still a guess and should be measured before fabrication. The separate `/api/v1/concept-mesh` endpoint still creates only a fixed stepped-cornice placeholder from supplied dimensions.

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

## Offline whitefill STL tool

The backend now also exposes a local whitefill tool for classroom tests that do not need internet or API keys:

- Open `http://127.0.0.1:8000/whitefill`
- Upload or capture 1-8 photos
- Click **Make STL** to generate one STL per detected piece

This workflow targets one piece colour per scan (white by default; `colour=yellow|red|orange|green|blue|pink` for solid coloured pieces). `fill=complete` restores broken cubes and triangles and gives other shapes a best-guess outline; each broken piece also yields a fixed photo (missing parts painted in). Marker print and result downloads are available from:

- `GET /api/v1/whitefill/marker`
- `POST /api/v1/whitefill/analyze`
- `GET /api/v1/whitefill/files/{batch}/{scan}/{name}`
- `GET /api/v1/whitefill/scans` (lists saved results)

## Rebel support and Archive

The dashboard's Rebel view sends support questions and a limited diagnostic snapshot to the configured Ollama service. Rebel suggests steps, but can make mistakes; it cannot inspect or edit source files, click controls, or run commands. Conversation history stays in the browser until the page is reloaded. Avoid sending secrets. Install the default support model once with `ollama pull qwen2.5:1.5b` (on VS Code Flatpak, use `flatpak-spawn --host ollama pull qwen2.5:1.5b`). `REBEL_MODEL` can select another installed Ollama text model independently of the image-analysis `OLLAMA_MODEL`. CPU inference can be slow or time out.

Archive lists locally saved Whitefill STL files, measurement sheets and fixed photos. Hosted generator downloads and converted GLBs are not automatically archived.

## Tests

```sh
python -m unittest discover -s backend/tests -v
```