# Tiny Autoencoder

Tiny Autoencoder is an interactive, full-stack demo of a CIFAR-10 autoencoder. Upload any image to see it resized to 32 × 32, compressed from 3,072 RGB values into a 512-value latent code, a 6× smaller representation,and reconstructed as a new 32 × 32 image. The interface visualizes each stage of the encoder, latent space, and decoder while the model processes the image.

Since the model was trained on the limited CIFAR-10 dataset, out-of-distribution images may produce imperfect reconstructions.

The project keeps the application layers separate:

- `frontend/` — dependency-free static web interface and processing animations.
- `backend/` — image validation, preprocessing, ONNX inference, and FastAPI routes.
- `app.py` — entry point that exposes the FastAPI application.

## Run locally

Create a virtual environment, install the lightweight inference dependencies, and start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

In another terminal, serve the frontend:

```powershell
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173`. During local development, the frontend sends reconstruction requests to `http://127.0.0.1:8000/api/reconstruct`.

## Model export

`backend/models/autoencoder_latent_512.onnx` is the inference model used by the API. It is an ONNX export of the original `latent_512.pt` checkpoint, so running the application does not require PyTorch. To regenerate it after retraining, install `torch` and `onnx`, then run:

```powershell
python -m backend.export_model --checkpoint "..\Frontier-AI\04_AutoEncoder\outputs\models\latent_512.pt"
```
