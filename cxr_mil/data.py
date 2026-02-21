from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def add_study_id(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a study_id column that uniquely identifies a (Patient_ID, Study) pair."""
    df = df.copy()
    df["study_id"] = df["Patient_ID"].astype(str) + "_" + df["Study"].astype(str)
    return df


def make_patient_level_folds(df: pd.DataFrame, label_cols: List[str], n_splits: int = 5, seed: int = 42):
    """
    Patient-level multilabel-stratified folds (so no patient leakage).
    Returns list of dicts: {fold, train_df, val_df}
    """
    patient_df = df.groupby("Patient_ID", as_index=False)[label_cols].max()
    X = patient_df[["Patient_ID"]].to_numpy()
    Y = patient_df[label_cols].to_numpy().astype(int)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_dfs = []
    for fold, (tr_idx, va_idx) in enumerate(mskf.split(X, Y), 1):
        train_pids = patient_df.loc[tr_idx, "Patient_ID"].values
        val_pids = patient_df.loc[va_idx, "Patient_ID"].values

        train_df = df[df["Patient_ID"].isin(train_pids)].reset_index(drop=True)
        val_df = df[df["Patient_ID"].isin(val_pids)].reset_index(drop=True)

        fold_dfs.append({"fold": fold, "train_df": train_df, "val_df": val_df})

    return fold_dfs


def compute_pos_weight_study_level(train_df_fold: pd.DataFrame, label_cols: List[str]) -> np.ndarray:
    """Study-level pos_weight = neg/pos per label (clipped)."""
    train_df_fold = add_study_id(train_df_fold)
    train_studies = train_df_fold.drop_duplicates("study_id").reset_index(drop=True)
    y = train_studies[label_cols].to_numpy(np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return np.clip(neg / (pos + 1e-6), 1.0, 50.0).astype(np.float32)


class StudyDataset(Dataset):
    """
    Returns one item per study (bag):
      imgs:  Tensor [K, 3, H, W]
      views: Tensor [K]  (0=frontal, 1=lateral)
      y:     Tensor [C]  (study-level labels)
      sid:   str
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, label_cols: List[str], transform: Optional[T.Compose] = None):
        self.df = add_study_id(df)
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.transform = transform

        # group by study_id (store row indices for fast lookup)
        groups = self.df.groupby("study_id", sort=False).indices
        self.study_ids = list(groups.keys())
        self._groups = groups

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx: int):
        sid = self.study_ids[idx]

        idxs = self._groups[sid]
        g = self.df.iloc[idxs]

        # Study label: take it from the first row in the study.
        # Assumption: study-level labels are consistent across rows within the same study.
        y = torch.tensor(g.iloc[0][self.label_cols].to_numpy(np.float32), dtype=torch.float32)

        imgs = []
        views = []
        for _, r in g.iterrows():
            path = os.path.join(self.img_dir, r["Image_name"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found for study_id={sid}: {path}")
            with Image.open(path) as img_pil:
                im = img_pil.convert("RGB")

            if self.transform is not None:
                im = self.transform(im)
            else:
                im = T.ToTensor()(im)

            vc = str(r.get("ViewCategory", "")).lower()
            view_idx = 0 if "front" in vc else 1  # non-frontal -> lateral bucket
            imgs.append(im)
            views.append(view_idx)

        imgs = torch.stack(imgs, dim=0)                     # [K, 3, H, W]
        views = torch.tensor(views, dtype=torch.long)        # [K]
        return imgs, views, y, sid


def mil_collate_fn(batch):
    """
    MIL collate (flatten instances across studies).

    Input batch items:
      (imgs[K,3,H,W], views[K], y[C], sid)

    Output:
      all_imgs:      [N,3,H,W]  concatenated instances across studies
      study_targets: [B,C]      one target per study
      group_idx:     [N]        image -> study index (0..B-1)
      view_idx:      [N]        view per image
      study_ids:     list[str]
    """
    all_imgs = []
    group_idx = []
    view_idx = []
    study_targets = []
    study_ids = []

    for b, (imgs, views, y, sid) in enumerate(batch):
        k = imgs.size(0)
        all_imgs.append(imgs)
        view_idx.append(views)
        group_idx.append(torch.full((k,), b, dtype=torch.long))
        study_targets.append(y)
        study_ids.append(sid)

    all_imgs = torch.cat(all_imgs, dim=0)
    view_idx = torch.cat(view_idx, dim=0)
    group_idx = torch.cat(group_idx, dim=0)
    study_targets = torch.stack(study_targets, dim=0)
    return all_imgs, study_targets, group_idx, view_idx, study_ids


def default_transforms():
    # Kept explicit and close to what you had.
    train_tf = T.Compose([
        T.Resize((512, 512)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=7),
        T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_loaders(
    train_df_fold: pd.DataFrame,
    val_df_fold: pd.DataFrame,
    img_dir: str,
    label_cols: List[str],
    study_batch_size: int,
    num_workers: int,
    train_tf,
    val_tf,
):
    train_ds = StudyDataset(train_df_fold, img_dir, label_cols, transform=train_tf)
    val_ds = StudyDataset(val_df_fold, img_dir, label_cols, transform=val_tf)

    dl_train_kwargs = dict(
        batch_size=study_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=mil_collate_fn,
    )
    dl_val_kwargs = dict(
        batch_size=study_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=mil_collate_fn,
    )

    if num_workers > 0:
        dl_train_kwargs["prefetch_factor"] = 2
        dl_val_kwargs["prefetch_factor"] = 2

    return DataLoader(train_ds, **dl_train_kwargs), DataLoader(val_ds, **dl_val_kwargs)
