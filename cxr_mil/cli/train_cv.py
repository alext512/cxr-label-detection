from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from ..config import TrainConfig
from ..data import add_study_id, build_loaders, compute_pos_weight_study_level, default_transforms, make_patient_level_folds
from ..io import load_dataframe
from ..losses import AsymmetricLoss
from ..metrics import mean_auc
from ..models import build_backbone, build_head, get_backbone_feature_module, load_backbone_weights
from ..train_steps import eval_one_epoch, predict_proba_study_loader, train_one_epoch
from ..transforms import build_transforms
from ..utils import set_seed


def _load_yaml(path: str) -> Dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out or {}


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_sd = model.state_dict()
        for key, val in model_sd.items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(val.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key] = val.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train CXR MIL model with patient-level cross-validation.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
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
    loss_cfg = yaml_cfg.get("loss", {}) or {}
    sched_cfg = yaml_cfg.get("scheduler", {}) or {}
    ema_cfg = yaml_cfg.get("ema", {}) or {}

    cfg = TrainConfig(
        root=data_cfg.get("root", "") or "",
        csv_name=data_cfg.get("csv_name", "train1.csv"),
        img_subdir=data_cfg.get("img_subdir", "train1"),
        run_name=exp_cfg.get("run_name", "densenet512_attn_posw_cv_v1"),
        seed=exp_cfg.get("seed", 42),
        output_root=exp_cfg.get("output_root", "runs"),
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
        max_grad_norm=train_cfg.get("max_grad_norm", None),
        num_workers=train_cfg.get("num_workers", 8),
        augmentations=aug_cfg,
        loss_name=loss_cfg.get("name", "weighted_bce"),
        pos_weight_clip=loss_cfg.get("pos_weight_clip", 50.0),
        asl_gamma_neg=loss_cfg.get("gamma_neg", 4.0),
        asl_gamma_pos=loss_cfg.get("gamma_pos", 1.0),
        asl_clip=loss_cfg.get("clip", 0.05),
        asl_eps=loss_cfg.get("eps", 1e-8),
        scheduler_name=sched_cfg.get("name", "reduce_on_plateau"),
        scheduler_factor=sched_cfg.get("factor", 0.5),
        scheduler_patience=sched_cfg.get("patience", 2),
        warmup_steps=sched_cfg.get("warmup_steps", 0),
        min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.05),
        use_ema=ema_cfg.get("enabled", False),
        ema_decay=ema_cfg.get("decay", 0.999),
        analysis=yaml_cfg.get("analysis", {}) or {},
    )

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
    return cfg


def _build_criterion(cfg: TrainConfig, pos_weight_np: np.ndarray, device: torch.device):
    if cfg.loss_name == "asl":
        return AsymmetricLoss(
            gamma_neg=cfg.asl_gamma_neg,
            gamma_pos=cfg.asl_gamma_pos,
            clip=cfg.asl_clip,
            eps=cfg.asl_eps,
        )
    if cfg.loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if cfg.loss_name == "weighted_bce":
        pw = np.clip(pos_weight_np, 1.0, float(cfg.pos_weight_clip)).astype(np.float32)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))
    raise ValueError(f"Unknown loss.name={cfg.loss_name}")


def _build_scheduler(cfg: TrainConfig, optimizer, total_optimizer_steps: int):
    if cfg.scheduler_name == "cosine_warmup":
        warmup = int(cfg.warmup_steps)
        min_ratio = float(cfg.min_lr_ratio)

        def lr_lambda(current_step: int):
            if warmup > 0 and current_step < warmup:
                return float(current_step + 1) / float(max(1, warmup))
            progress_steps = max(1, total_optimizer_steps - warmup)
            progress = min(1.0, max(0.0, (current_step - warmup) / progress_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"

    if cfg.scheduler_name == "reduce_on_plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.scheduler_factor),
            patience=int(cfg.scheduler_patience),
        )
        return sched, "epoch"

    raise ValueError(f"Unknown scheduler.name={cfg.scheduler_name}")


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    df, label_cols, img_dir = load_dataframe(cfg.root, cfg.csv_name, cfg.img_subdir)
    train_tf, val_tf = build_transforms(cfg.augmentations) if cfg.augmentations else default_transforms()
    folds = make_patient_level_folds(df, label_cols, n_splits=cfg.n_splits, seed=cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = Path(cfg.output_root) / cfg.run_name / str(cfg.seed)
    run_root.mkdir(parents=True, exist_ok=True)

    resolved_cfg = asdict(cfg)
    with open(run_root / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2)

    cv_log_path = run_root / "cv_training_log.csv"
    with open(cv_log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["fold", "epoch", "phase", "train_loss", "val_loss", "mean_auc", "lrs", "best_auc_so_far"])

    studies_df = add_study_id(df).drop_duplicates("study_id").reset_index(drop=True)
    id_to_row = {sid: i for i, sid in enumerate(studies_df["study_id"].tolist())}
    oof_true = studies_df[label_cols].to_numpy(np.int64)
    oof_pred = np.full((len(studies_df), len(label_cols)), np.nan, dtype=np.float32)
    fold_best: List[float] = []
    fold_rows: List[Dict[str, Any]] = []

    for fd in folds:
        fold = fd["fold"]
        if cfg.select_fold is not None and fold != cfg.select_fold:
            continue

        fold_dir = run_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(resolved_cfg, f, indent=2)

        train_df_fold, val_df_fold = fd["train_df"], fd["val_df"]
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

        backbone, emb_dim = build_backbone(cfg.backbone_name, pretrained=cfg.backbone_pretrained, **(cfg.backbone_kwargs or {}))
        backbone = backbone.to(device)
        if cfg.backbone_checkpoint_path:
            load_backbone_weights(
                backbone,
                cfg.backbone_checkpoint_path,
                checkpoint_key=cfg.backbone_checkpoint_key,
                prefix_strip=cfg.backbone_prefix_strip,
                strict=cfg.backbone_load_strict,
                adapt_input_conv=cfg.backbone_adapt_input_conv,
            )

        study_head = build_head(cfg.head_name, emb_dim=emb_dim, num_classes=len(label_cols), **(cfg.head_kwargs or {})).to(device)
        criterion = _build_criterion(cfg, pos_weight_np, device)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        feature_module = get_backbone_feature_module(backbone)

        def build_optimizer(frozen: bool):
            for p in feature_module.parameters():
                p.requires_grad = not frozen
            if frozen:
                return torch.optim.AdamW(study_head.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
            return torch.optim.AdamW(
                [{"params": feature_module.parameters(), "lr": cfg.backbone_lr}, {"params": study_head.parameters(), "lr": cfg.head_lr}],
                weight_decay=cfg.weight_decay,
            )

        start_frozen = (cfg.frozen_epochs or 0) > 0
        optimizer = build_optimizer(frozen=start_frozen)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, cfg.accum_steps)))
        scheduler, sched_mode = _build_scheduler(cfg, optimizer, total_optimizer_steps=steps_per_epoch * cfg.total_epochs)

        ema_backbone = ModelEMA(backbone, cfg.ema_decay) if cfg.use_ema else None
        ema_head = ModelEMA(study_head, cfg.ema_decay) if cfg.use_ema else None

        best_auc = -1.0
        best_epoch = 0
        best_path = fold_dir / "best.pt"

        fold_log_path = fold_dir / "training_log.csv"
        with open(fold_log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["fold", "epoch", "phase", "train_loss", "val_loss", "mean_auc", "lrs", "best_auc_so_far"])

        for epoch in range(1, cfg.total_epochs + 1):
            if start_frozen and epoch == cfg.frozen_epochs + 1:
                optimizer = build_optimizer(frozen=False)
                scheduler, sched_mode = _build_scheduler(cfg, optimizer, total_optimizer_steps=steps_per_epoch * cfg.total_epochs)

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
                max_grad_norm=cfg.max_grad_norm,
                on_optimizer_step=(scheduler.step if sched_mode == "step" else None),
            )

            if cfg.use_ema and ema_backbone is not None and ema_head is not None:
                ema_backbone.update(backbone)
                ema_head.update(study_head)
                eval_backbone, eval_head = build_backbone(cfg.backbone_name, pretrained=False, **(cfg.backbone_kwargs or {}))[0].to(device), build_head(cfg.head_name, emb_dim=emb_dim, num_classes=len(label_cols), **(cfg.head_kwargs or {})).to(device)
                eval_backbone.load_state_dict(backbone.state_dict())
                eval_head.load_state_dict(study_head.state_dict())
                ema_backbone.copy_to(eval_backbone)
                ema_head.copy_to(eval_head)
            else:
                eval_backbone, eval_head = backbone, study_head

            va_loss = eval_one_epoch(eval_backbone, eval_head, val_loader, criterion, device)
            if sched_mode == "epoch":
                scheduler.step(va_loss)
            lrs = [pg["lr"] for pg in optimizer.param_groups]

            val_ids, val_probs = predict_proba_study_loader(eval_backbone, eval_head, val_loader, device)
            val_studies_df = add_study_id(val_df_fold).drop_duplicates("study_id").reset_index(drop=True)
            sid_to_true = dict(zip(val_studies_df["study_id"], val_studies_df[label_cols].to_numpy(np.int64)))
            y_true = np.stack([sid_to_true[sid] for sid in val_ids], axis=0)
            m_auc, _ = mean_auc(y_true, val_probs, label_cols)

            if m_auc > best_auc:
                best_auc = m_auc
                best_epoch = epoch
                save_backbone = eval_backbone if eval_backbone is not backbone else backbone
                save_head = eval_head if eval_head is not study_head else study_head
                torch.save({"backbone": save_backbone.state_dict(), "study_head": save_head.state_dict()}, best_path)

            row = [fold, epoch, phase, tr_loss, va_loss, m_auc, str(lrs), best_auc]
            with open(fold_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            with open(cv_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            if eval_backbone is not backbone:
                del eval_backbone, eval_head

        fold_best.append(best_auc)
        fold_rows.append({"fold": fold, "best_epoch": best_epoch, "best_auc": float(best_auc), "best_checkpoint": str(best_path)})

        ckpt = torch.load(best_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        study_head.load_state_dict(ckpt["study_head"])

        val_ids, val_probs = predict_proba_study_loader(backbone, study_head, val_loader, device)
        val_studies_df = add_study_id(val_df_fold).drop_duplicates("study_id").reset_index(drop=True)
        fold_oof_df = val_studies_df[["study_id", "Patient_ID", "Study"] + label_cols].copy()
        for i, col in enumerate(label_cols):
            fold_oof_df[f"pred_{col}"] = val_probs[:, i]
        fold_oof_df.to_csv(fold_dir / "oof_predictions.csv", index=False)

        for sid, prob in zip(val_ids, val_probs):
            oof_pred[id_to_row[sid]] = prob

        del backbone, study_head, optimizer, scheduler, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "run_dir": str(run_root),
        "fold_best_aucs": [float(x) for x in fold_best],
        "mean_auc": float(np.mean(fold_best)) if fold_best else float("nan"),
        "std_auc": float(np.std(fold_best)) if fold_best else float("nan"),
        "fold_details": fold_rows,
    }

    with open(run_root / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if np.isfinite(oof_pred).all():
        oof_mean, per_label = mean_auc(oof_true, oof_pred, label_cols)
        oof_df = studies_df[["study_id", "Patient_ID", "Study"] + label_cols].copy()
        for i, col in enumerate(label_cols):
            oof_df[f"pred_{col}"] = oof_pred[:, i]
        oof_df.to_csv(run_root / "oof_predictions_all_folds.csv", index=False)

        with open(run_root / "oof_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"oof_mean_auc": float(oof_mean), "per_label_auc": per_label}, f, indent=2)


if __name__ == "__main__":
    main()
