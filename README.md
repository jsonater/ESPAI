# Robo Rebels · ESPAI

Heritage imaging and reconstruction research prototype.

## Start the app

GitHub provides the files, but does not run the Python backend. Each computer
needs Python 3.10 or newer and an internet connection for the initial install.
Clone or download and extract this repository, then open a terminal in the
`ESPAI` folder.

On Windows (PowerShell or Command Prompt):

```bat
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Leave the server terminal open and visit **http://127.0.0.1:8000/** on the
same computer. Don't open `index.html` directly or use a GitHub-hosted static
page: those do not start the backend. On later visits, run only the last
command for your operating system. If startup fails, check the terminal error;
if port 8000 is occupied, stop the other server or choose a free port and use
that port in the browser. The car camera also requires connecting that computer
to the car's Wi-Fi. Ollama and hosted 3D generation have separate requirements
described below.

- `index.html`: local image collection and labeling interface.
- `ai/`: image-level condition classifier training starter.
- `backend/`: ESP32-CAM `/Camera` WebSocket receiver, localhost MJPEG dashboard API, local Ollama and optional Roboflow image analysis, hosted single-image 3D concept workflow with local GLB-to-STL conversion, local whitefill STL tool (`/whitefill`), and dimensioned concept-STL API.

The backend receives binary JPEG frames from the car's WebSocket while the laptop is on the car Wi-Fi. Roboflow Cloud and the hosted Stable Fast 3D generator require internet. Stable Fast 3D can generate a plausible mesh from one photo; it does not guarantee a repaired or historically accurate object, hidden geometry, real-world scale, or print-ready quality. The separate dimensioned STL route remains a parametric placeholder, not image reconstruction.

See [backend/README.md](backend/README.md) for camera/API setup and [ai/README.md](ai/README.md) for the image-classifier training workflow.