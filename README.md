# Saliency IQA

Saliency IQA predicts the perceived quality of an image. It combines a Swin-B global branch with a saliency U-Net that guides a VGG16 local branch, fuses the two with channel self-attention, and regresses a patch-weighted quality score. A web app shows the score, a quality label and the model's saliency map.

## Features

- Quality score for any uploaded image, on the scale the model was trained on
- Quality label: Excellent, Good, Fair, Poor or Bad
- Saliency heatmap showing where the model looks
- Web app, REST API and command-line scoring
- Training and evaluation scripts for your own dataset
- CPU and NVIDIA GPU Docker images

## Quick Start

Put a trained checkpoint at `checkpoints/bestnatmodel.pth`, then build and run:

```bash
docker compose up --build web
```

Open http://localhost:8000 (API docs at http://localhost:8000/docs).

To run on an NVIDIA GPU (served on port 8080):

```bash
docker compose --profile gpu up --build web-gpu
```

The GPU image needs the NVIDIA Container Toolkit (on Windows: Docker Desktop with WSL2 and a recent NVIDIA driver).

## Tech Stack

- **Model:** PyTorch, timm (Swin-B), torchvision (VGG16)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, JavaScript
- **Data:** Pillow, NumPy, pandas, SciPy

## API Endpoints

- `GET /`: web app
- `POST /api/predict`: upload an image and get its score (add `?saliency=false` to skip the heatmap)
- `GET /api/health`: health check, device and score range

Example:

```bash
curl -F "file=@photo.jpg" http://localhost:8000/api/predict
```

```json
{"score": 3.87, "normalized": 0.7175, "label": "Good", "scale_min": 1, "scale_max": 5,
 "saliency_png": "<base64>", "inference_ms": 85.2}
```

## Training

Put images in `data/images/` and a `data/labels.csv` with `Image,Quality` columns:

```
Image,Quality
I01_01_01.png,4.57
I01_01_02.png,4.33
```

Train with Docker (GPU):

```bash
docker compose --profile train run --rm train
```

Or locally:

```bash
python train.py --images data/images --csv data/labels.csv --score-min 1 --score-max 5
```

| Setting | Value |
|---|---|
| Split | 80 / 10 / 10 train / val / test, seed 42 |
| Optimizer | Adam, lr 1e-4, weight decay 1e-6, batch 32 |
| Backbones | frozen for the first 20 epochs, then fine-tuned |
| Loss | MSE + 0.4 · (1 − PLCC) + 0.1 · rank loss |
| Model selection | best validation PLCC, early stopping with patience 10 |

Training writes `bestnatmodel.pth`, a per-epoch log and the final test metrics to `checkpoints/`. Use `--image-col` and `--score-col` for other CSV headers, `--amp` for mixed precision, and `--num-workers 0` if the DataLoader hangs on Windows.

Common score ranges:

| Dataset | Range | Flags |
|---|---|---|
| KADID-10k, KonIQ-10k | 1 – 5 | `--score-min 1 --score-max 5` |
| TID2013 | 0 – 9 | `--score-min 0 --score-max 9` |
| LIVE-C, SPAQ | 0 – 100 | `--score-min 0 --score-max 100` |

## Command Line

Score images or a folder without running the app:

```bash
python predict.py photo.jpg
python predict.py some_folder/
```

Evaluate a checkpoint (PLCC, SRCC, RMSE):

```bash
python evaluate.py --checkpoint checkpoints/bestnatmodel.pth --split test
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `checkpoints/bestnatmodel.pth` | checkpoint to serve |
| `SAL_INPUT` | from checkpoint, else `image` | saliency U-Net input: `image` or `salimg` |
| `DEVICE` | `cuda` if available | force `cpu` or `cuda` |
| `MAX_UPLOAD_MB` | `20` | upload size limit |
| `TORCH_THREADS` | number of CPU cores | CPU inference threads |

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Requirements

- Docker
- Python 3.11+ for local development
- A GPU is optional for inference and recommended for training
