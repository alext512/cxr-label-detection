from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def mean_auc(y_true: np.ndarray, y_pred: np.ndarray, label_cols: List[str]) -> Tuple[float, dict]:
    """
    Computes per-label AUC when possible; ignores labels with only one class present in y_true.
    """
    aucs = {}
    for i, name in enumerate(label_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs[name] = float(roc_auc_score(yt, yp))
        except Exception:
            continue

    if len(aucs) == 0:
        return float("nan"), {}
    return float(np.mean(list(aucs.values()))), aucs
