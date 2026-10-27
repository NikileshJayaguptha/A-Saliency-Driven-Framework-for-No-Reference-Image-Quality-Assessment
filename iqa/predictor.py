import base64
import io

import numpy as np
import torch
from PIL import Image

from .data import preprocess
from .model import SaliencyIQA

# Defaults for checkpoints saved by the notebook (no metadata stored):
# KADID-10k DMOS in [1,5], normalized as (q-1)/4.
DEFAULT_META = {"score_min": 1.0, "score_max": 5.0, "sal_input": "image"}


def _label(norm):
    if norm >= 0.8: return "Excellent"
    if norm >= 0.6: return "Good"
    if norm >= 0.4: return "Fair"
    if norm >= 0.2: return "Poor"
    return "Bad"


def _heatmap_png(sal, pil_image, size=448):
    """Overlay a min-max normalized saliency map on the image, return base64 PNG."""
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    base = pil_image.convert("RGB").resize((size, size))
    heat = Image.fromarray((sal * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    h = np.asarray(heat, dtype=np.float32) / 255.0
    # simple warm colormap: dark -> red -> yellow
    rgb = np.stack([np.clip(h * 2, 0, 1), np.clip(h * 2 - 1, 0, 1), np.zeros_like(h)], -1)
    out = (0.55 * np.asarray(base, dtype=np.float32) / 255.0 + 0.45 * rgb)
    buf = io.BytesIO()
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Predictor:
    def __init__(self, checkpoint_path, device=None, sal_input=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = {**DEFAULT_META, **ckpt.get("meta", {})}
        if sal_input:
            meta["sal_input"] = sal_input
        self.meta = meta
        # pretrained=False: every weight comes from the checkpoint, nothing is downloaded
        self.model = SaliencyIQA(pretrained=False, sal_input=meta["sal_input"])
        self.model.load_checkpoint_dict(ckpt)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, pil_image, with_saliency=True):
        salimg, image = preprocess(pil_image)
        salimg, image = salimg[None].to(self.device), image[None].to(self.device)
        score, saliency, patch_scores, patch_weights = self.model(image, salimg, return_maps=True)
        norm = float(score.item())
        lo, hi = self.meta["score_min"], self.meta["score_max"]
        norm_c = min(max(norm, 0.0), 1.0)
        result = {
            "score": round(lo + norm_c * (hi - lo), 4),
            "score_raw": round(lo + norm * (hi - lo), 4),
            "normalized": round(norm, 4),
            "scale_min": lo,
            "scale_max": hi,
            "label": _label(norm_c),
        }
        if with_saliency:
            result["saliency_png"] = _heatmap_png(saliency[0, 0].float().cpu().numpy(), pil_image)
        return result
