"""
Train the saliency-guided IQA model.

    python train.py --images data/images --csv data/labels.csv

Same recipe as the notebook: 80/10/10 split (seed 42), batch 32, Adam lr 1e-4,
wd 1e-6, backbones frozen for the first 20 epochs, loss = MSE + 0.4*PLCC + 0.1*rank,
early stopping on validation PLCC (patience 10). Best weights are written in the
same format as the notebook's bestnatmodel.pth, plus a 'meta' entry.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from iqa.data import IQADataset, split_dataset
from iqa.losses import combined_loss, metrics
from iqa.model import SaliencyIQA


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default="data/images", help="folder with the images")
    p.add_argument("--csv", default="data/labels.csv", help="CSV with image names and scores")
    p.add_argument("--image-col", default="Image")
    p.add_argument("--score-col", default="Quality")
    p.add_argument("--score-min", type=float, default=1.0, help="lowest possible score in the CSV")
    p.add_argument("--score-max", type=float, default=5.0, help="highest possible score in the CSV")
    p.add_argument("--out", default="checkpoints/bestnatmodel.pth")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--freeze-epochs", type=int, default=20, help="epochs with backbones frozen")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sal-input", choices=["image", "salimg"], default="image",
                   help="input to the saliency U-Net (notebook trained with 'image')")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--no-pretrained", action="store_true", help="random init (offline testing only)")
    p.add_argument("--limit", type=int, default=0, help="use only N samples (smoke test)")
    return p.parse_args()


def set_requires_grad(modules, flag):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for salimg, image, mos in tqdm(loader, desc="eval", leave=False):
        score = model(image.to(device), salimg.to(device))
        preds.extend(score.view(-1).float().cpu().numpy())
        labels.extend(mos.view(-1).numpy())
    return metrics(preds, labels)


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = IQADataset(a.images, a.csv, a.image_col, a.score_col, a.score_min, a.score_max)
    if a.limit:
        ds = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    train_ds, val_ds, test_ds = split_dataset(ds, seed=a.seed)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > a.batch_size, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    model = SaliencyIQA(pretrained=not a.no_pretrained, sal_input=a.sal_input).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")
    meta = {"score_min": a.score_min, "score_max": a.score_max, "sal_input": a.sal_input}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    log_path = os.path.splitext(a.out)[0] + "_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_plcc", "val_srcc", "val_rmse"])

    best, counter = -float("inf"), 0
    for epoch in range(a.epochs):
        frozen = epoch < a.freeze_epochs
        set_requires_grad(model.backbones(), not frozen)
        set_requires_grad(model.heads(), True)

        model.train()
        total = 0.0
        for salimg, image, mos in tqdm(train_loader, desc=f"Epoch {epoch+1}{' (frozen)' if frozen else ''}"):
            salimg, image, mos = salimg.to(device), image.to(device), mos.to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                score = model(image, salimg)
            loss = combined_loss(score.float(), mos)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        train_loss = total / max(len(train_loader), 1)

        m = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | loss {train_loss:.4f} | val PLCC {m['PLCC']:.4f} "
              f"SRCC {m['SRCC']:.4f} RMSE {m['RMSE']:.4f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, m["PLCC"], m["SRCC"], m["RMSE"]])

        if m["PLCC"] > best:
            best, counter = m["PLCC"], 0
            torch.save(model.checkpoint_dict({**meta, "epoch": epoch + 1, "val": m}), a.out)
            print(f"  best model saved -> {a.out}")
        else:
            counter += 1
            print(f"  patience {counter}/{a.patience}")
            if counter >= a.patience:
                print("Early stopping.")
                break

    # final test on the held-out 10% with the best weights
    model.load_checkpoint_dict(torch.load(a.out, map_location="cpu", weights_only=False))
    model.to(device)
    t = evaluate(model, test_loader, device)
    print("TEST  " + "  ".join(f"{k}: {v:.4f}" for k, v in t.items()))
    with open(os.path.splitext(a.out)[0] + "_test.json", "w") as f:
        json.dump(t, f, indent=2)


if __name__ == "__main__":
    main()
