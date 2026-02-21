from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import NON_LABEL_COLS
from ..metrics import mean_auc


def parse_args():
    p = argparse.ArgumentParser(description="Analyze OOF predictions and produce error reports.")
    p.add_argument("--oof", required=True, help="Path to OOF predictions CSV (from training run dir).")
    p.add_argument("--out-dir", default="analysis_reports", help="Directory where analysis CSVs are written.")
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.oof)
    label_cols = [c for c in df.columns if c not in NON_LABEL_COLS + ["study_id"] and not c.startswith("pred_")]
    pred_cols = [f"pred_{c}" for c in label_cols]

    y_true = df[label_cols].to_numpy(np.int64)
    y_pred = df[pred_cols].to_numpy(np.float32)
    mean_cv_auc, per_label_auc = mean_auc(y_true, y_pred, label_cols)

    metrics_df = pd.DataFrame({"label": label_cols, "auc": [per_label_auc.get(c, np.nan) for c in label_cols]})
    metrics_df.sort_values("auc", ascending=True).to_csv(out_dir / "per_label_auc.csv", index=False)

    eps = 1e-6
    bce = -(y_true * np.log(np.clip(y_pred, eps, 1 - eps)) + (1 - y_true) * np.log(np.clip(1 - y_pred, eps, 1 - eps)))
    df["sample_bce"] = bce.mean(axis=1)

    worst = df.sort_values("sample_bce", ascending=False).head(args.top_k)
    best = df.sort_values("sample_bce", ascending=True).head(args.top_k)
    worst.to_csv(out_dir / "worst_predicted_studies.csv", index=False)
    best.to_csv(out_dir / "best_predicted_studies.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "oof_mean_auc", "value": mean_cv_auc},
        {"metric": "n_studies", "value": len(df)},
        {"metric": "top_k", "value": args.top_k},
    ])
    summary.to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
