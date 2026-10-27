"""
Web service: upload an image, get a quality score.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from iqa.predictor import Predictor

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/bestnatmodel.pth")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
STATIC = Path(__file__).parent / "static"

state = {}


@asynccontextmanager
async def lifespan(app):
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Put bestnatmodel.pth in ./checkpoints "
                           f"or set MODEL_PATH.")
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 1)))
    t = time.time()
    state["predictor"] = Predictor(MODEL_PATH, device=os.getenv("DEVICE") or None,
                                   sal_input=os.getenv("SAL_INPUT") or None)
    print(f"Model loaded from {MODEL_PATH} on {state['predictor'].device} in {time.time()-t:.1f}s")
    yield
    state.clear()


app = FastAPI(title="Saliency-guided Image Quality Assessment", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    p = state.get("predictor")
    return {"status": "ok" if p else "loading", "device": str(p.device) if p else None,
            "scale": [p.meta["score_min"], p.meta["score_max"]] if p else None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), saliency: bool = True):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB:g} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Not a readable image file")
    t = time.time()
    result = state["predictor"].predict(img, with_saliency=saliency)
    result["inference_ms"] = round((time.time() - t) * 1000, 1)
    result["filename"] = file.filename
    result["width"], result["height"] = img.size
    return result
