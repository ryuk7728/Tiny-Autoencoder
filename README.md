# Neural Rebuild

An interactive, full-stack demo of the trained CIFAR-10 latent-512 autoencoder. Upload an image, see the exact 32 × 32 RGB input, and compare it with the model's reconstruction.

The project keeps the user-facing layers separate:

- `frontend/` — dependency-free static web app and processing animation.
- `backend/` — image validation/preprocessing, ONNX model, and FastAPI routes.
- `app.py` — small Vercel entrypoint that exposes the backend as one function.

## What the 6× figure means

Every upload is first converted to RGB and resized to 32 × 32. That is `32 × 32 × 3 = 3,072` numeric colour values. The encoder maps those to a 512-value latent vector, which is six times fewer values. The decoder turns that vector into a *new* 32 × 32 image.

This is a learning demo, not a high-resolution image compressor or upscaler: original detail is discarded at resize time. Because the checkpoint was trained on CIFAR-10, images unlike small natural-object photos may reconstruct imperfectly.

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

Open `http://localhost:5173`. The frontend automatically calls `http://127.0.0.1:8000/api/reconstruct` in local development and `/api/reconstruct` once deployed.

## Model export

`backend/models/autoencoder_latent_512.onnx` is the production model. It is an ONNX export of the original `latent_512.pt` checkpoint, so production does not need to ship PyTorch. To regenerate it after retraining, install `torch` and `onnx`, then run:

```powershell
python -m backend.export_model --checkpoint "..\Frontier-AI\04_AutoEncoder\outputs\models\latent_512.pt"
```

## Deploy to Vercel

The static frontend is built into `public/`; Vercel serves it from its CDN. `app.py` is detected as the FastAPI application and exposes the `/api/*` routes in the same deployment.

```powershell
npm install
npx vercel
npx vercel --prod
```

The deployment configuration explicitly includes the ONNX model for the Python function. The API accepts PNG, JPG, and WebP uploads up to 4 MB, processes them in memory, and does not write uploads to disk.

