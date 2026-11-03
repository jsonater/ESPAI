# Vestige image classifier

This is the first AI component: a small image-level classifier for `Missing section`, `Fracture`, `Surface erosion`, and `Intact reference`. It is not a restoration model, does not draw damage masks, and does not generate 3D geometry.

The trainer consumes a JSON bundle exported by the **Training image library** in `index.html`. It creates a small 64 x 64 RGB CNN, evaluates it on a held-out image split, and exports a full-int8 TensorFlow Lite model plus C headers for a later ESP32 firmware integration.

## Prepare images

- Collect rights-cleared, real facade photos and verify labels before saving them.
- Use JPEG or PNG images. Convert phone HEIC photos to JPEG first.
- Collect at least 10 images for each of at least two condition labels to run the baseline. This is only a minimum for exercising the pipeline; credible generalization requires substantially more diverse images.
- Include varied sites, viewpoints, lighting, distances, and intact reference examples. Avoid putting near-duplicate photos in the collection.
- In the website, choose images, set their condition and site, add them to the dataset, then select **Export data**. Keep the JSON bundle private; it contains the original images.

The current split is stratified by label but random by image. It can put images from the same site in both training and test sets, so reported test accuracy may overstate performance on unseen sites. A site-held-out evaluation should be added before treating metrics as research results.

## Train

Linux Mint, from the project directory:

```sh
source .venv/bin/activate
python -m pip install -r ai/requirements.txt
python ai/train_classifier.py /path/to/vestige-training-dataset.json
```

Outputs are written to `ai/output/`:

- `condition_classifier_int8.tflite`: fully quantized model with signed int8 inputs and outputs.
- `model_data.h`: model bytes as a C++ array.
- `model_labels.h`: class names in output-index order.
- `condition_classifier.keras`: editable Keras model.
- `metrics.json`: image counts, split counts, test accuracy, and model size.

Training is local on the computer, not on the ESP32. The headers are ingredients for firmware, not a complete sketch. A microcontroller deployment still needs TensorFlow Lite Micro-compatible firmware and a successful memory/operator compatibility check for the exact board. The model expects 64 x 64 RGB values normalized to 0..1 before quantization; firmware must use the input tensor's quantization scale and zero point.

## Tests

The dataset validation tests use only the Python standard library:

```sh
python -m unittest discover -s ai/tests
```