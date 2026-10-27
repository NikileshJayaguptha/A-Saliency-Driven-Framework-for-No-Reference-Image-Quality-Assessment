"""
Score one or more images from the command line.

    python predict.py photo.jpg another.png
    python predict.py some_folder/ --checkpoint checkpoints/bestnatmodel.pth
"""
import argparse
import os

from PIL import Image

from iqa.predictor import Predictor

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--checkpoint", default=os.getenv("MODEL_PATH", "checkpoints/bestnatmodel.pth"))
    p.add_argument("--sal-input", choices=["image", "salimg"], default=None)
    a = p.parse_args()

    files = []
    for path in a.paths:
        if os.path.isdir(path):
            files += sorted(os.path.join(path, f) for f in os.listdir(path)
                            if os.path.splitext(f)[1].lower() in EXTS)
        else:
            files.append(path)

    pred = Predictor(a.checkpoint, sal_input=a.sal_input)
    for f in files:
        r = pred.predict(Image.open(f), with_saliency=False)
        print(f"{f}\t{r['score']:.3f} / {r['scale_max']:g}\t({r['label']}, normalized {r['normalized']:.3f})")


if __name__ == "__main__":
    main()
