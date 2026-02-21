from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

from .constants import EXPECTED_LABEL_COLS


NON_LABEL_COLS = ["Image_name","Patient_ID","Study","Sex","Age","ViewCategory","ViewPosition"]


def load_dataframe(root: str, csv_name: str = "train1.csv", img_subdir: str = "train1") -> Tuple[pd.DataFrame, List[str], str]:
    csv_path = os.path.join(root, csv_name)
    img_dir  = os.path.join(root, img_subdir)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    df = pd.read_csv(csv_path)

    required_meta_cols = {"Image_name", "Patient_ID", "Study"}
    missing_meta = required_meta_cols - set(df.columns)
    if missing_meta:
        raise ValueError(f"CSV is missing required metadata columns: {sorted(missing_meta)}")

    label_cols = [c for c in df.columns if c not in NON_LABEL_COLS]
    missing = set(EXPECTED_LABEL_COLS) - set(label_cols)
    extra   = set(label_cols) - set(EXPECTED_LABEL_COLS)
    if missing or extra:
        raise ValueError(f"Label mismatch. Missing={missing}, Extra={extra}")

    return df, list(EXPECTED_LABEL_COLS), img_dir
