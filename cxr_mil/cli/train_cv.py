from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import TrainConfig
from ..utils import set_seed
from ..io import load_dataframe
from ..data import (
    make_patient_level_folds,
    compute_pos_weight_study_level,
    build_loaders,
    default_transforms,
)
from ..models import build_backbone, build_head, load_backbone_weights, get_backbone_feature_module
from ..train_steps import train_one_epoch, eval_one_epoch, predict_proba_study_loader
from ..metrics import mean_auc
from ..transforms import build_transforms



def _load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config (PyYAML required)."""
    if path is None:
        return {}

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required to read config files.\n"
            "Install it with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        out = yaml.safe_load(f)

    return out or {}



def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train CXR MIL model with patient-level cross-validation.",
    )
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (overrides defaults).")
    p.add_argument("--root", type=str, default=None, help="Dataset root containing CSV and data.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--frozen-epochs", type=int, default=None)
    p.add_argument("--study-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--select-fold", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--n-splits", type=int, default=None)
    args = p.parse_args()

    yaml_cfg = _load_yaml(args.config)
    train_cfg = yaml_cfg.get("train", {})
    data_cfg = yaml_cfg.get("data", {})
    cv_cfg = yaml_cfg.get("cv", {})
    exp_cfg = yaml_cfg.get("experiment", {})
    aug_cfg = yaml_cfg.get("augmentations", {}) or {}

    model_cfg = yaml_cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    head_cfg = model_cfg.get("head", {})

    # Build config with YAML overrides
    cfg = TrainConfig(
        root=data_cfg.get("root", "") or "",
        csv_name=data_cfg.get("csv_name", "train1.csv"),
        img_subdir=data_cfg.get("img_subdir", "train1"),
        run_name=exp_cfg.get("run_name", "densenet512_attn_posw_cv_v1"),
        seed=exp_cfg.get("seed", 42),
        backbone_name=backbone_cfg.get("name", "densenet121"),
        backbone_pretrained=backbone_cfg.get("pretrained", True),
        backbone_kwargs=backbone_cfg.get("kwargs", {}) or {},
        backbone_checkpoint_path=backbone_cfg.get("checkpoint_path", None),
        backbone_checkpoint_key=backbone_cfg.get("checkpoint_key", None),
        backbone_prefix_strip=backbone_cfg.get("prefix_strip", None),
        backbone_load_strict=backbone_cfg.get("load_strict", False),
        backbone_adapt_input_conv=backbone_cfg.get("adapt_input_conv", "auto"),
        head_name=head_cfg.get("name", "view_aware_gated_attn"),
        head_kwargs=head_cfg.get("kwargs", {}) or {},
        n_splits=cv_cfg.get("n_splits", 5),
        select_fold=cv_cfg.get("select_fold", None),
        total_epochs=train_cfg.get("total_epochs", 20),
        frozen_epochs=train_cfg.get("frozen_epochs", 2),
        head_lr=train_cfg.get("head_lr", 3e-4),
        backbone_lr=train_cfg.get("backbone_lr", 1e-5),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        accum_steps=train_cfg.get("accum_steps", 4),
        study_batch_size=train_cfg.get("study_batch_size", 48),
        num_workers=train_cfg.get("num_workers", 8),
        augmentations=aug_cfg,
    )

    # CLI overrides (highest priority)
    if args.root is not None:
        cfg.root = args.root
    if args.epochs is not None:
        cfg.total_epochs = args.epochs
    if args.frozen_epochs is not None:
        cfg.frozen_epochs = args.frozen_epochs
    if args.study_batch_size is not None:
        cfg.study_batch_size = args.study_batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.select_fold is not None:
        cfg.select_fold = args.select_fold
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.n_splits is not None:
        cfg.n_splits = args.n_splits

    if not cfg.root:
        raise ValueError("Dataset root is required. Provide --root or set data.root in YAML config.")
    if cfg.n_splits < 2:
        raise ValueError("cv.n_splits must be >= 2.")
    if cfg.select_fold is not None and not (1 <= cfg.select_fold <= cfg.n_splits):
        raise ValueError(f"cv.select_fold must be in [1, {cfg.n_splits}] or null.")

    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    df, label_cols, img_dir = load_dataframe(cfg.root, cfg.csv_name, cfg.img_subdir)

    if cfg.augmentations:
        train_tf, val_tf = build_transforms(cfg.augmentations)
    else:
        train_tf, val_tf = default_transforms()

    folds = make_patient_level_folds(df, label_cols, n_splits=cfg.n_splits, seed=cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Config:", {k: v for k, v in asdict(cfg).items() if k not in {"root"}})
    print("Root:", cfg.root)

    log_path = f"{cfg.run_name}_training_log_cv.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "epoch", "phase", "train_loss", "val_loss", "mean_auc", "lrs", "best_auc_so_far"])

    # OOF tracking at study level
    df_sid = df.copy()
    df_sid["study_id"] = df_sid["Patient_ID"].astype(str) + "_" + df_sid["Study"].astype(str)
    studies_df = df_sid.drop_duplicates("study_id").reset_index(drop=True)
    id_to_row = {sid: i for i, sid in enumerate(studies_df["study_id"].tolist())}
    oof_true = studies_df[label_cols].to_numpy(np.int64)
    oof_pred = np.full((len(studies_df), len(label_cols)), np.nan, dtype=np.float32)

    fold_best: List[float] = []

    for fd in folds:
        fold = fd["fold"]
        if cfg.select_fold is not None and fold != cfg.select_fold:
            continue

        print("\n" + "=" * 80)
        print(f"FOLD {fold}/{cfg.n_splits}")
        print("=" * 80)

        train_df_fold = fd["train_df"]
        val_df_fold = fd["val_df"]

        pos_weight_np = compute_pos_weight_study_level(train_df_fold, label_cols)

        train_loader, val_loader = build_loaders(
            train_df_fold,
            val_df_fold,
            img_dir,
            label_cols,
            study_batch_size=cfg.study_batch_size,
            num_workers=cfg.num_workers,
            train_tf=train_tf,
            val_tf=val_tf,
        )

        backbone, emb_dim = build_backbone(
            cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            **(cfg.backbone_kwargs or {}),
        )
        backbone = backbone.to(device)

        if cfg.backbone_checkpoint_path:
            rep = load_backbone_weights(
                backbone,
                cfg.backbone_checkpoint_path,
                checkpoint_key=cfg.backbone_checkpoint_key,
                prefix_strip=cfg.backbone_prefix_strip,
                strict=cfg.backbone_load_strict,
                adapt_input_conv=cfg.backbone_adapt_input_conv,
            )
            print("Loaded custom backbone weights:", rep)

        study_head = build_head(
            cfg.head_name,
            emb_dim=emb_dim,
            num_classes=len(label_cols),
            **(cfg.head_kwargs or {}),
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
        )

        # Use the non-deprecated API (keeps behavior the same)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        def build_optimizer_and_scheduler(frozen: bool):
            # Freeze/unfreeze backbone feature extractor
            feature_module = get_backbone_feature_module(backbone)
            for p in feature_module.parameters():
                p.requires_grad = (not frozen)

            # During frozen phase: only train the head
            if frozen:
                opt = torch.optim.AdamW(
                    study_head.parameters(),
                    lr=cfg.head_lr,
                    weight_decay=cfg.weight_decay,
                )
            # During unfrozen phase: train backbone + head with different LRs
            else:
                opt = torch.optim.AdamW(
                    [
                        {"params": feature_module.parameters(), "lr": cfg.backbone_lr},
                        {"params": study_head.parameters(), "lr": cfg.head_lr},
                    ],
                    weight_decay=cfg.weight_decay,
                )

            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=2
            )
            return opt, sched

        # If frozen_epochs == 0, start unfrozen immediately.
        start_frozen = (cfg.frozen_epochs is not None) and (cfg.frozen_epochs > 0)
        optimizer, scheduler = build_optimizer_and_scheduler(frozen=start_frozen)

        best_auc = -1.0
        best_epoch = 0
        best_path = f"{cfg.run_name}_fold{fold}_best.pt"

        for epoch in range(1, cfg.total_epochs + 1):
            # Switch from frozen -> unfrozen exactly once, right after frozen_epochs.
            if start_frozen and epoch == cfg.frozen_epochs + 1:
                optimizer, scheduler = build_optimizer_and_scheduler(frozen=False)

            phase = "frozen" if (start_frozen and epoch <= cfg.frozen_epochs) else "unfrozen"

            tr_loss = train_one_epoch(
                backbone,
                study_head,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                accum_steps=cfg.accum_steps,
            )
            va_loss = eval_one_epoch(backbone, study_head, val_loader, criterion, device)
            scheduler.step(va_loss)
            lrs = [pg["lr"] for pg in optimizer.param_groups]

            # Predict validation probabilities at study level.
            val_ids, val_probs = predict_proba_study_loader(backbone, study_head, val_loader, device)

            # align true labels by study_id
            val_sid_df = val_df_fold.copy()
            val_sid_df["study_id"] = val_sid_df["Patient_ID"].astype(str) + "_" + val_sid_df["Study"].astype(str)
            val_studies_df = val_sid_df.drop_duplicates("study_id").reset_index(drop=True)
            sid_to_true = dict(zip(val_studies_df["study_id"], val_studies_df[label_cols].to_numpy(np.int64)))
            missing_ids = [sid for sid in val_ids if sid not in sid_to_true]
            if missing_ids:
                raise KeyError(
                    f"Missing {len(missing_ids)} validation study IDs in ground truth mapping. "
                    f"Example: {missing_ids[0]}"
                )
            y_true = np.stack([sid_to_true[sid] for sid in val_ids], axis=0)

            m_auc, _ = mean_auc(y_true, val_probs, label_cols)

            if m_auc > best_auc:
                best_auc = m_auc
                best_epoch = epoch
                torch.save({"backbone": backbone.state_dict(), "study_head": study_head.state_dict()}, best_path)
                print(f"[best] Fold {fold}: saved at epoch {epoch} ({phase}) | AUC {m_auc:.4f}")

            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([fold, epoch, phase, tr_loss, va_loss, m_auc, str(lrs), best_auc])

            print(
                f"Fold {fold} | Epoch {epoch:02d} ({phase}) | "
                f"train {tr_loss:.4f} | val {va_loss:.4f} | auc {m_auc:.4f} | LRs {lrs}"
            )

        print(f"\nFold {fold} BEST: epoch {best_epoch} | AUC {best_auc:.4f} | saved: {best_path}")
        fold_best.append(best_auc)

        # Fill OOF preds for this fold using best ckpt
        ckpt = torch.load(best_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        study_head.load_state_dict(ckpt["study_head"])

        # Predict OOF with the best checkpoint for this fold.
        val_ids, val_probs = predict_proba_study_loader(backbone, study_head, val_loader, device)
        for sid, prob in zip(val_ids, val_probs):
            oof_pred[id_to_row[sid]] = prob

        del backbone, study_head, optimizer, scheduler, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("CV DONE")
    print("=" * 80)
    if fold_best:
        print("Fold best AUCs:", [round(x, 4) for x in fold_best])
        print(f"Mean AUC: {float(np.mean(fold_best)):.4f}  |  Std: {float(np.std(fold_best)):.4f}")
    print(f"Log saved to: {log_path}")

    if np.isfinite(oof_pred).all():
        oof_mean, _ = mean_auc(oof_true, oof_pred, label_cols)
        print(f"OOF mean AUC: {oof_mean:.4f}")
    else:
        print("OOF predictions not fully filled (expected if you trained only one fold).")


if __name__ == "__main__":
    main()
