"""Single-image preprocessing and real inference utilities for AgroVision AI."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src import config
from src.data_loader import format_class_label
from src.evaluate import load_class_names


class InvalidImageError(ValueError):
    """Raised when an uploaded or supplied file cannot be decoded as a valid image."""


DISEASE_INFO: dict[str, dict[str, str]] = {
    "Tomato_Bacterial_spot": {
        "cause": "Bacterium (Xanthomonas species)",
        "summary": (
            "Causes small, dark, water-soaked spots on leaves that may turn greasy, "
            "coalesce, and develop yellow halos, leading to defoliation in warm, humid weather."
        ),
        "guidance": (
            "Inspect neighboring plants, avoid overhead watering to reduce splash spread, "
            "and consult an agronomist before applying copper-based treatments."
        ),
    },
    "Tomato_Early_blight": {
        "cause": "Fungus (Alternaria solani)",
        "summary": (
            "Characterized by dark brown lesions with concentric target-like rings, "
            "typically starting on older lower leaves and often surrounded by yellowing tissue."
        ),
        "guidance": (
            "Remove heavily affected lower leaves, improve airflow and mulching, "
            "and verify management options with local agricultural guidance."
        ),
    },
    "Tomato_Late_blight": {
        "cause": "Oomycete pathogen (Phytophthora infestans)",
        "summary": (
            "Produces large, irregular, water-soaked pale-green to dark-brown blotches "
            "that spread rapidly in cool, wet conditions, sometimes with white mold underneath."
        ),
        "guidance": (
            "Late blight spreads quickly; isolate severely infected foliage promptly "
            "and seek immediate advice from a qualified agricultural extension specialist."
        ),
    },
    "Tomato_Leaf_Mold": {
        "cause": "Fungus (Passalora fulva / Fulvia fulva)",
        "summary": (
            "Appears as pale-green or yellowish patches on upper leaf surfaces paired "
            "with olive-green to velvety gray-purple spore growth on the underside."
        ),
        "guidance": (
            "Common in high-humidity greenhouses; increase ventilation, reduce leaf wetness, "
            "and confirm treatment steps with an agronomist."
        ),
    },
    "Tomato_healthy": {
        "cause": "No disease detected",
        "summary": (
            "The leaf displays uniform green coloration without visible necrotic lesions, "
            "chlorotic halos, or fungal/bacterial spotting characteristic of the trained disease classes."
        ),
        "guidance": (
            "Continue routine monitoring, balanced irrigation, and preventive field hygiene."
        ),
    },
}


def load_model_and_classes(
    model_path: Path = config.MODEL_PATH,
    class_names_path: Path = config.CLASS_NAMES_PATH,
) -> tuple[tf.keras.Model, list[str]]:
    """Load the saved Keras model and ordered class label list."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    class_names = load_class_names(class_names_path)
    model = tf.keras.models.load_model(model_path)
    return model, class_names


def _read_raw_bytes(image_source: Path | str | bytes | BinaryIO) -> bytes:
    """Extract raw bytes from a path, bytes object, or binary file-like upload."""
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.is_file():
            raise InvalidImageError(f"Image file does not exist: {path}")
        return path.read_bytes()
    if isinstance(image_source, (bytes, bytearray)):
        return bytes(image_source)
    if hasattr(image_source, "read"):
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        data = image_source.read()
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        return data
    raise InvalidImageError("Unsupported image input type.")


def validate_and_load_image(
    image_source: Path | str | bytes | BinaryIO,
) -> Image.Image:
    """Validate an image stream/file via Pillow and return an orientation-corrected RGB image."""
    raw_bytes = _read_raw_bytes(image_source)
    if not raw_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            corrected = ImageOps.exif_transpose(img)
            return corrected.convert("RGB")
    except Exception as exc:
        raise InvalidImageError(
            "Unable to decode a valid JPG, JPEG, or PNG leaf image."
        ) from exc


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 and apply MobileNetV2 [-1, 1] normalization once."""
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(batch)


def predict_image(
    image_source: Path | str | bytes | BinaryIO | Image.Image,
    model: tf.keras.Model | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Run real MobileNetV2 inference on a single leaf image."""
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()

    if isinstance(image_source, Image.Image):
        pil_image = image_source.convert("RGB")
    else:
        pil_image = validate_and_load_image(image_source)

    input_batch = preprocess_pil_image(pil_image)
    probs = model.predict(input_batch, verbose=0)[0]

    if len(probs) != len(class_names):
        raise ValueError(
            f"Model output size ({len(probs)}) does not match class count ({len(class_names)})."
        )

    ranked_indices = np.argsort(probs)[::-1]
    top_idx = int(ranked_indices[0])
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    ranked_predictions = [
        {
            "class_name": class_names[int(idx)],
            "display_name": format_class_label(class_names[int(idx)]),
            "probability": float(probs[int(idx)]),
            "percentage": float(probs[int(idx)] * 100.0),
        }
        for idx in ranked_indices
    ]

    info = DISEASE_INFO.get(
        predicted_class,
        {
            "cause": "Unknown",
            "summary": "No description available for this class.",
            "guidance": "Consult an agricultural specialist for confirmation.",
        },
    )

    return {
        "predicted_class": predicted_class,
        "display_name": format_class_label(predicted_class),
        "confidence": confidence,
        "confidence_pct": confidence * 100.0,
        "probabilities": ranked_predictions,
        "disease_info": info,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_leaf_image>")
        raise SystemExit(1)

    output = predict_image(Path(sys.argv[1]))
    print(json.dumps(output, indent=2))
