# Put your dataset here

```
data/
├── images/          <- all image files (any of jpg/png/bmp/...)
│   ├── I01_01_01.png
│   ├── I01_01_02.png
│   └── ...
└── labels.csv       <- one row per image
```

`labels.csv` needs two columns (names can be changed with `--image-col` / `--score-col`):

```
Image,Quality
I01_01_01.png,4.57
I01_01_02.png,4.33
```

`Image` is the file name relative to `data/images/`. `Quality` is the MOS/DMOS.

Tell the trainer the score range with `--score-min` / `--score-max`
(scores are normalized to 0–1 for training and mapped back for display):

| Dataset   | range     | flags                              |
|-----------|-----------|------------------------------------|
| KADID-10k | 1 – 5     | `--score-min 1 --score-max 5` (default) |
| TID2013   | 0 – 9     | `--score-min 0 --score-max 9`      |
| KonIQ-10k | 1 – 5     | `--score-min 1 --score-max 5`      |
| LIVE-C    | 0 – 100   | `--score-min 0 --score-max 100`    |
| SPAQ      | 0 – 100   | `--score-min 0 --score-max 100`    |

KADID's original `dmos.csv` uses columns `dist_img,ref_img,dmos`; either rename them
to `Image,Quality` or pass `--image-col dist_img --score-col dmos`.
