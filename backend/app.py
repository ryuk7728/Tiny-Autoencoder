"""HTTP API for the interactive autoencoder demo."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.inference import ImageInputError, MAX_UPLOAD_BYTES, reconstruct


logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app = FastAPI(title="Tiny Autoencoder API", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ready", "model": "autoencoder-latent-512"}


@app.post("/api/reconstruct")
async def reconstruct_image(image: UploadFile = File(...)) -> dict[str, object]:
    """Accept an image and return its 32×32 input plus model reconstruction."""
    contents = await image.read(MAX_UPLOAD_BYTES + 1)
    try:
        return reconstruct(contents)
    except ImageInputError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        logger.exception("Reconstruction model failed to load or run")
        raise HTTPException(status_code=503, detail="The reconstruction model is warming up. Try again shortly.") from error


# Keep this mount last so the API routes above take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
