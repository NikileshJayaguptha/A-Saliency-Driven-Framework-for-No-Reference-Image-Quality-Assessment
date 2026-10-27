# Saliency-guided Image Quality Assessment — end-to-end

Upload an image → get a predicted quality score (plus the model's saliency map).
Architecture is the one from `saliency unet try-Kadid.ipynb`:
Swin-B (global) + saliency U-Net → saliency-modulated VGG16 (local) → channel
self-attention → patch-wise weighted regressor.

```
iqa-app/
├── iqa/                 # model, dataset, losses, predictor (importable package)
│   ├── model.py         # SaliencyIQA – loads notebook checkpoints directly
│   ├── data.py          # dataset + 80/10/10 split (seed 42, same as notebook)
│   ├── losses.py        # MSE + 0.4·PLCC + 0.1·rank, metrics
│   └── predictor.py     # single-image inference + saliency overlay
├── app/                 # FastAPI website
│   ├── main.py          # GET /  ·  POST /api/predict  ·  GET /api/health
│   └── static/index.html
├── train.py             # training (same recipe as the notebook)
├── evaluate.py          # PLCC / SRCC / RMSE on train/val/test/all
├── predict.py           # command-line scoring
├── data/                # ← your dataset goes here (see data/README.md)
│   ├── images/
│   └── labels.csv
├── checkpoints/         # ← put bestnatmodel.pth here
├── Dockerfile
└── docker-compose.yml
```

---

## 1. Use the model you already trained

Copy your notebook's `bestnatmodel.pth` into `checkpoints/`. Nothing else to convert.

### With Docker (recommended)

```bash
docker compose up --build web                    # CPU
docker compose --profile gpu up --build web-gpu  # NVIDIA GPU
```

Open **http://localhost:8000**, drop an image, read the score.

GPU needs the NVIDIA Container Toolkit (on Windows: Docker Desktop + WSL2 + a recent NVIDIA driver).

### Without Docker

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows  (Linux/mac: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Command line:

```bash
python predict.py my_photo.jpg
python predict.py some_folder/
```

API:

```bash
curl -F "file=@my_photo.jpg" http://localhost:8000/api/predict
# {"score": 3.87, "normalized": 0.7175, "label": "Good", "scale_min": 1, "scale_max": 5,
#  "saliency_png": "<base64>", "inference_ms": 85.2, ...}
```
Add `?saliency=false` to skip the heatmap.

---

## 2. Train (or retrain) from scratch

1. Put images in `data/images/` and a CSV in `data/labels.csv` with columns `Image,Quality`
   (details and per-dataset score ranges in `data/README.md`).
2. Run training.

**Docker (GPU):**
```bash
docker compose --profile train run --rm train
# custom flags:
docker compose --profile train run --rm train python train.py \
    --images /app/data/images --csv /app/data/labels.csv \
    --out /app/checkpoints/bestnatmodel.pth --score-min 0 --score-max 9
```

**Local:**
```bash
python train.py --images data/images --csv data/labels.csv --score-min 1 --score-max 5
```

What it does (identical to the notebook):
- 80 / 10 / 10 split with seed 42
- batch 32, Adam lr 1e-4, weight decay 1e-6
- epochs 1–20: Swin, VGG and saliency U-Net frozen; only attention + regressor learn
- epoch 21+: everything fine-tuned
- loss = MSE + 0.4 · (1 − PLCC) + 0.1 · rank loss
- best model by validation PLCC, early stopping with patience 10
- at the end, evaluates the best weights on the held-out test 10%

Outputs in `checkpoints/`: `bestnatmodel.pth`, `bestnatmodel_log.csv` (per-epoch metrics),
`bestnatmodel_test.json` (final test metrics). The `.pth` also stores the score range so the
website shows scores in the right scale automatically.

Useful flags: `--epochs`, `--freeze-epochs`, `--patience`, `--batch-size` (lower it if you run
out of GPU memory), `--amp` (mixed precision, ~1.5–2× faster on GPU), `--num-workers`
(use `0` on Windows if DataLoader hangs), `--image-col/--score-col` for other CSV headers.

Re-check a model at any time:
```bash
python evaluate.py --checkpoint checkpoints/bestnatmodel.pth --split test
```

---

## 3. Two things to know about your notebook

**Saliency input mismatch.** During training and validation the notebook fed the saliency
U-Net the 224 px ImageNet-normalized `image`, but the final test cell fed it `salimg`
(256 px, un-normalized). The 0.9702 test PLCC came from that second path. This project
defaults to `image` (what the model was trained on). Check which works better on your test split:

```bash
python evaluate.py --split test --sal-input image
python evaluate.py --split test --sal-input salimg
```
and, if `salimg` wins, run the site with `SAL_INPUT=salimg` (add it under `environment:` in
`docker-compose.yml`).

**timm version.** Swin layer names changed between timm 0.6 and 0.9+. If loading fails on the
`swin` part, run `pip show timm` in your notebook environment and pin that version in
`requirements.txt`.

---

## Configuration (environment variables)

| Variable        | Default                              | Meaning                          |
|-----------------|--------------------------------------|----------------------------------|
| `MODEL_PATH`    | `checkpoints/bestnatmodel.pth`       | checkpoint to serve              |
| `SAL_INPUT`     | from checkpoint, else `image`        | `image` or `salimg`              |
| `DEVICE`        | auto (`cuda` if available)           | force `cpu` / `cuda`             |
| `MAX_UPLOAD_MB` | `20`                                 | upload size limit                |
| `TORCH_THREADS` | number of CPU cores                  | CPU inference threads            |

Checkpoints saved by the notebook have no metadata, so they are assumed to be KADID-style
scores in 1–5 (normalized as `(q − 1) / 4`). Models trained with `train.py` carry their own range.
