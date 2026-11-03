"""Train a small, quantized facade-condition classifier from a Vestige export."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONDITIONS = (
    "Missing section",
    "Fracture",
    "Surface erosion",
    "Intact reference",
)
IMAGE_SIZE = 64


@dataclass(frozen=True)
class Sample:
    image_bytes: bytes
    condition: str
    site: str
    file_name: str


def decode_data_url(data_url: str, file_name: str) -> bytes:
    header, separator, payload = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError(f"{file_name}: image is not a base64-encoded image data URL")
    if not (header.startswith("data:image/jpeg") or header.startswith("data:image/png")):
        raise ValueError(f"{file_name}: use a JPEG or PNG image (phone HEIC files must be converted first)")
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{file_name}: image data is not valid base64") from error
    if not image_bytes:
        raise ValueError(f"{file_name}: image data is empty")
    return image_bytes


def load_samples(path: Path) -> list[Sample]:
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read dataset JSON: {error}") from error
    if bundle.get("format") != "vestige-dataset-v1":
        raise ValueError("Unsupported dataset format; export data from the Vestige dataset page")
    records = bundle.get("samples")
    if not isinstance(records, list):
        raise ValueError("Dataset bundle must contain a 'samples' list")

    samples: list[Sample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Sample {index + 1} is not an object")
        condition = record.get("condition")
        if condition == "Unknown":
            continue
        if condition not in CONDITIONS:
            raise ValueError(
                f"Sample {record.get('fileName', index + 1)} has unsupported condition {condition!r}"
            )
        file_name = str(record.get("fileName", f"sample-{index + 1}"))
        data_url = record.get("imageDataUrl")
        if not isinstance(data_url, str):
            raise ValueError(f"{file_name}: missing embedded image data")
        samples.append(
            Sample(
                image_bytes=decode_data_url(data_url, file_name),
                condition=condition,
                site=str(record.get("site", "Unknown")),
                file_name=file_name,
            )
        )
    if not samples:
        raise ValueError("No usable labeled images found; Unknown condition labels are skipped")
    return samples


def validate_class_counts(samples: list[Sample], minimum: int) -> list[str]:
    counts = Counter(sample.condition for sample in samples)
    classes = [condition for condition in CONDITIONS if counts[condition] > 0]
    if len(classes) < 2:
        raise ValueError("At least two different visible-condition labels are needed to train a classifier")
    insufficient = {condition: counts[condition] for condition in classes if counts[condition] < minimum}
    if insufficient:
        details = ", ".join(f"{label}: {count}" for label, count in insufficient.items())
        raise ValueError(
            f"Need at least {minimum} images per condition for a basic train/validation/test split; "
            f"currently below minimum: {details}"
        )
    return classes


def stratified_indices(samples: list[Sample], classes: list[str], seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for condition in classes:
        indices = [index for index, sample in enumerate(samples) if sample.condition == condition]
        rng.shuffle(indices)
        test_count = max(1, round(len(indices) * 0.15))
        validation_count = max(1, round(len(indices) * 0.15))
        test.extend(indices[:test_count])
        validation.extend(indices[test_count:test_count + validation_count])
        train.extend(indices[test_count + validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def write_c_header(path: Path, model_bytes: bytes) -> None:
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "",
        "alignas(16) const unsigned char g_vestige_model[] = {",
    ]
    for offset in range(0, len(model_bytes), 12):
        chunk = model_bytes[offset:offset + 12]
        lines.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    lines.extend([
        "};",
        f"const unsigned int g_vestige_model_len = {len(model_bytes)};",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_labels_header(path: Path, classes: list[str]) -> None:
    labels = ",\n".join(f"  {c_string(label)}" for label in classes)
    path.write_text(
        "#pragma once\n"
        f"constexpr int g_vestige_label_count = {len(classes)};\n"
        f"const char* const g_vestige_labels[g_vestige_label_count] = {{\n{labels}\n}};\n",
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow and NumPy are required. Install them with: "
            "python -m pip install -r ai/requirements.txt"
        ) from error

    samples = load_samples(args.dataset)
    classes = validate_class_counts(samples, args.min_per_class)
    class_to_index = {condition: index for index, condition in enumerate(classes)}
    train_indices, validation_indices, test_indices = stratified_indices(samples, classes, args.seed)

    images = np.empty((len(samples), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.empty((len(samples),), dtype=np.int32)
    for index, sample in enumerate(samples):
        decoded = tf.io.decode_image(sample.image_bytes, channels=3, expand_animations=False)
        decoded = tf.image.resize(decoded, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
        images[index] = tf.cast(decoded, tf.float32).numpy() / 255.0
        labels[index] = class_to_index[sample.condition]

    tf.keras.utils.set_random_seed(args.seed)
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax", name="condition")(x)
    model = tf.keras.Model(inputs, outputs, name="vestige_condition_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        images[train_indices],
        labels[train_indices],
        validation_data=(images[validation_indices], labels[validation_indices]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(images[test_indices], labels[test_indices], verbose=0)
    model.save(output_dir / "condition_classifier.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [images[index:index + 1]] for index in train_indices[:min(len(train_indices), 100)]
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    model_path = output_dir / "condition_classifier_int8.tflite"
    model_path.write_bytes(tflite_model)
    write_c_header(output_dir / "model_data.h", tflite_model)
    write_labels_header(output_dir / "model_labels.h", classes)

    metrics = {
        "model": "Vestige 64x64 condition classifier",
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "input_normalization": "RGB values divided by 255",
        "quantization": "full int8",
        "labels": classes,
        "image_count": len(samples),
        "class_counts": dict(Counter(sample.condition for sample in samples)),
        "split_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_validation_accuracy": max(history.history.get("val_accuracy", [0.0])),
        "split_warning": "Images are split randomly, not by site; results may overestimate performance on new sites.",
        "model_bytes": len(tflite_model),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if len(tflite_model) > 512 * 1024:
        print("Warning: model exceeds 512 KiB; it may not fit the target ESP32 board's available memory.", file=sys.stderr)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Vestige JSON bundle exported from the dataset page")
    parser.add_argument("--output", type=Path, default=Path("ai/output"), help="Directory for model files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.min_per_class < 2:
        parser.error("epochs and batch size must be positive; min-per-class must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    try:
        samples = load_samples(args.dataset)
        counts = Counter(sample.condition for sample in samples)
        print(f"Loaded {len(samples)} labeled JPEG/PNG images: {dict(counts)}")
        metrics = train(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 1
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Quantized model: {args.output / 'condition_classifier_int8.tflite'} ({metrics['model_bytes']} bytes)")
    print(f"ESP32 headers: {args.output / 'model_data.h'} and {args.output / 'model_labels.h'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())