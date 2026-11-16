# Robo Rebels · ESPAI

Heritage imaging and reconstruction research prototype.

- `index.html`: local image collection and labeling interface.
- `ai/`: image-level condition classifier training starter.
- `backend/`: ESP32-CAM `/Camera` WebSocket receiver, localhost MJPEG dashboard API, local Ollama and optional Roboflow image analysis, and dimensioned concept-STL API.

The backend receives binary JPEG frames from the car's WebSocket while the laptop is on the car Wi-Fi. Roboflow Cloud requires internet, so frames can be captured locally and analyzed after reconnecting to an internet network. The STL route creates a parametric placeholder mesh from supplied measurements; it does not infer a missing component from a photograph. Generated geometry is not approved for restoration or 3D printing.

See [backend/README.md](backend/README.md) for camera/API setup and [ai/README.md](ai/README.md) for the image-classifier training workflow.