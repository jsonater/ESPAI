# Robo Rebels · ESPAI

Heritage imaging and reconstruction research prototype.

- `index.html`: local image collection and labeling interface.
- `ai/`: image-level condition classifier training starter.
- `backend/`: ESP32-CAM MJPEG reader and dimensioned concept-STL API.

The backend validates a camera image and creates a parametric placeholder mesh from supplied measurements. It does not infer a missing component from a photograph. Generated geometry is not approved for restoration or 3D printing.

See [backend/README.md](backend/README.md) for camera/API setup and [ai/README.md](ai/README.md) for the image-classifier training workflow.