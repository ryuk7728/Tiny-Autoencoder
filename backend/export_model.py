"""One-off utility that converts the trained PyTorch checkpoint to ONNX.

Torch is intentionally not a production dependency. The web API runs the
generated ONNX file with ONNX Runtime, which keeps the Vercel function small.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from backend.model import AutoEncoder


PROJECTS_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECTS_DIR
    / "Frontier-AI"
    / "04_AutoEncoder"
    / "outputs"
    / "models"
    / "latent_512.pt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "autoencoder_latent_512.onnx"


def export(checkpoint: Path, output: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = AutoEncoder(latent_size=512)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["image"],
        output_names=["reconstruction"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the latent-512 model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.checkpoint, args.output)

