"""
Evaluate a checkpoint on a labelled dataset.

    python evaluate.py --checkpoint checkpoints/bestnatmodel.pth --split test
    python evaluate.py --split all --images other_data/images --csv other_data/labels.csv
"""
import argparse

import torch
from torch.utils.data import DataLoader

from iqa.data import IQADataset, split_dataset
from iqa.model import SaliencyIQA
from iqa.predictor import DEFAULT_META
from train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/bestnatmodel.pth")
    p.add_argument("--images", default="data/images")
    p.add_argument("--csv", default="data/labels.csv")
    p.add_argument("--image-col", default="Image")
    p.add_argument("--score-col", default="Quality")
    p.add_argument("--score-min", type=float, default=None)
    p.add_argument("--score-max", type=float, default=None)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sal-input", choices=["image", "salimg"], default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = {**DEFAULT_META, **ckpt.get("meta", {})}
    lo = a.score_min if a.score_min is not None else meta["score_min"]
    hi = a.score_max if a.score_max is not None else meta["score_max"]

    model = SaliencyIQA(pretrained=False, sal_input=a.sal_input or meta["sal_input"])
    model.load_checkpoint_dict(ckpt)
    model.to(device)

    ds = IQADataset(a.images, a.csv, a.image_col, a.score_col, lo, hi)
    if a.split != "all":
        ds = dict(zip(["train", "val", "test"], split_dataset(ds, seed=a.seed)))[a.split]
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
    m = evaluate(model, loader, device)
    print(f"{a.split} ({len(ds)} images)  " + "  ".join(f"{k}: {v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
