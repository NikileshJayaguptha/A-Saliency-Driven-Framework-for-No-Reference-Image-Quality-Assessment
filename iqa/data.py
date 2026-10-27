import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

saliency_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def preprocess(pil_image):
    """PIL image -> (salimg, image) tensors without batch dim."""
    pil_image = pil_image.convert("RGB")
    return saliency_transform(pil_image), image_transform(pil_image)


class IQADataset(Dataset):
    """
    image_dir : folder with images
    csv_path  : CSV with an image-name column and a quality-score column
    Scores are normalized to [0,1] with (q - score_min) / (score_max - score_min).
    """

    def __init__(self, image_dir, csv_path, image_col="Image", score_col="Quality",
                 score_min=1.0, score_max=5.0):
        self.image_dir = image_dir
        df = pd.read_csv(csv_path)
        for c in (image_col, score_col):
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not in {csv_path}. Found: {list(df.columns)}")
        self.names = df[image_col].astype(str).tolist()
        self.scores = df[score_col].astype(float).tolist()
        self.score_min, self.score_max = float(score_min), float(score_max)

        missing = [n for n in self.names if not os.path.isfile(os.path.join(image_dir, n))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images listed in the CSV are not in {image_dir}, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.names[idx]))
        salimg, image = preprocess(img)
        mos = (self.scores[idx] - self.score_min) / (self.score_max - self.score_min)
        return salimg, image, torch.tensor(mos, dtype=torch.float32)


def split_dataset(dataset, seed=42, train_frac=0.8, val_frac=0.1):
    """Same 80/10/10 split as the notebook (same seed -> same split)."""
    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)
