# Robo Rebels · ESPAI

Heritage imaging and reconstruction research prototype.

- `index.html`: local image collection and labeling interface.
- `ai/`: image-level condition classifier training starter.
- `backend/`: ESP32-CAM `/Camera` WebSocket receiver, localhost MJPEG dashboard API, local Ollama and optional Roboflow image analysis, hosted single-image 3D concept workflow with local GLB-to-STL conversion, and dimensioned concept-STL API.

The backend receives binary JPEG frames from the car's WebSocket while the laptop is on the car Wi-Fi. Roboflow Cloud and the hosted Stable Fast 3D generator require internet. Stable Fast 3D can generate a plausible mesh from one photo; it does not guarantee a repaired or historically accurate object, hidden geometry, real-world scale, or print-ready quality. The separate dimensioned STL route remains a parametric placeholder, not image reconstruction.

See [backend/README.md](backend/README.md) for camera/API setup and [ai/README.md](ai/README.md) for the image-classifier training workflow.