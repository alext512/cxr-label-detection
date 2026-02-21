from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from ..io import load_dataframe
from ..transforms import build_transforms


def _load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_pil(tensor_img):
    if hasattr(tensor_img, "detach"):
        arr = tensor_img.detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)
    return tensor_img


def _make_grid(images, rows, cols):
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(im, (c * w, r * h))
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Render augmentation samples into an image artifact.")
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="analysis_reports/augmentation_samples.png")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    cfg = _load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentations", {})
    if not aug_cfg:
        raise ValueError("No augmentations block in config.")

    df, _, img_dir = load_dataframe(args.root, data_cfg.get("csv_name", "train1.csv"), data_cfg.get("img_subdir", "train1"))
    train_tf, val_tf = build_transforms(aug_cfg)

    sample_df = df.sample(n=min(args.num_samples, len(df)), random_state=args.seed).reset_index(drop=True)
    ims = []
    to_tensor = T.ToTensor()
    for _, row in sample_df.iterrows():
        img = Image.open(Path(img_dir) / row["Image_name"]).convert("RGB")
        train_out = _to_pil(train_tf(img))
        val_out = _to_pil(val_tf(img))
        raw_out = _to_pil(to_tensor(img))
        ims.extend([raw_out, train_out, val_out])

    grid = _make_grid(ims, rows=len(sample_df), cols=3)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


if __name__ == "__main__":
    main()
