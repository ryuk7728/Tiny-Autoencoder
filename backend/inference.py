"""Safe image preprocessing and ONNX inference for the public API."""

from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_PATH = Path(__file__).resolve().parent / "models" / "autoencoder_latent_512.onnx"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageInputError(ValueError):
    """An uploaded file cannot safely be used as an image."""


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    if not MODEL_PATH.is_file():
        raise RuntimeError("The ONNX model is missing from the deployment.")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def _to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _open_and_prepare(contents: bytes) -> tuple[Image.Image, tuple[int, int]]:
    if not contents:
        raise ImageInputError("Choose an image before reconstructing it.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ImageInputError("Please upload an image smaller than 4 MB.")

    try:
        with Image.open(BytesIO(contents)) as source:
            source.load()
            original_size = source.size
            # Respect phone-camera orientation before converting and resizing.
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ImageInputError("That file is not a supported, safely sized image.") from error

    reduced = image.resize((32, 32), Image.Resampling.LANCZOS)
    return reduced, original_size


def reconstruct(contents: bytes) -> dict[str, object]:
    """Run an upload through the trained 32×32 → 512 → 32×32 autoencoder."""
    reduced, original_size = _open_and_prepare(contents)
    image_array = np.asarray(reduced, dtype=np.float32) / 255.0
    model_input = np.transpose(image_array, (2, 0, 1))[None, ...]

    session = get_session()
    output_name = session.get_outputs()[0].name
    reconstructed = session.run([output_name], {"image": model_input})[0][0]
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    reconstructed_image = Image.fromarray(
        (np.transpose(reconstructed, (1, 2, 0)) * 255).round().astype(np.uint8),
        mode="RGB",
    )

    return {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "input_32x32": _to_data_url(reduced),
        "reconstruction": _to_data_url(reconstructed_image),
        "input_values": 32 * 32 * 3,
        "latent_values": 512,
        "compression_ratio": 6,
    }

