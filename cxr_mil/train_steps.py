from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm.auto import tqdm


def _freeze_batchnorm_stats(m: nn.Module) -> None:
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


def _unwrap_logits(head_out):
    return head_out[0] if isinstance(head_out, tuple) else head_out


def train_one_epoch(
    backbone: nn.Module,
    study_head: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler,
    accum_steps: int = 4,
    max_grad_norm: float | None = None,
    on_optimizer_step=None,
) -> float:
    backbone.train()
    study_head.train()
    backbone.apply(_freeze_batchnorm_stats)

    total_loss = 0.0
    total_studies = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="train", leave=False)

    last_step = 0
    for step, (all_imgs, study_targets, group_idx, view_idx, _) in enumerate(pbar, 1):
        last_step = step
        all_imgs = all_imgs.to(device, non_blocking=True)
        study_targets = study_targets.to(device, non_blocking=True)
        group_idx = group_idx.to(device, non_blocking=True)
        view_idx = view_idx.to(device, non_blocking=True)
        B = int(study_targets.size(0))

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            inst_embs = backbone(all_imgs)
            logits = _unwrap_logits(study_head(inst_embs, group_idx, view_idx, B))
            loss = criterion(logits, study_targets)

        scaler.scale(loss / accum_steps).backward()
        total_loss += float(loss.item()) * B
        total_studies += B
        pbar.set_postfix(loss=f"{(total_loss / max(1, total_studies)):.4f}")

        if step % accum_steps == 0:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(study_head.parameters()),
                    max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if on_optimizer_step is not None:
                on_optimizer_step()

    if last_step % accum_steps != 0:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(study_head.parameters()),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if on_optimizer_step is not None:
            on_optimizer_step()

    return total_loss / max(1, total_studies)


@torch.no_grad()
def eval_one_epoch(backbone: nn.Module, study_head: nn.Module, loader, criterion, device: torch.device) -> float:
    backbone.eval()
    study_head.eval()
    running = 0.0
    pbar = tqdm(loader, desc="val", leave=False)

    for step, (all_imgs, study_targets, group_idx, view_idx, _) in enumerate(pbar, 1):
        all_imgs = all_imgs.to(device, non_blocking=True)
        study_targets = study_targets.to(device, non_blocking=True)
        group_idx = group_idx.to(device, non_blocking=True)
        view_idx = view_idx.to(device, non_blocking=True)
        B = int(study_targets.size(0))

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            inst_embs = backbone(all_imgs)
            logits = _unwrap_logits(study_head(inst_embs, group_idx, view_idx, B))
            loss = criterion(logits, study_targets)

        running += float(loss.item())
        pbar.set_postfix(loss=f"{running/step:.4f}")

    return running / max(1, len(loader))


@torch.no_grad()
def predict_proba_study_loader(backbone: nn.Module, study_head: nn.Module, loader, device: torch.device):
    backbone.eval()
    study_head.eval()
    all_ids: list[str] = []
    all_probs: list[np.ndarray] = []

    for all_imgs, _, group_idx, view_idx, study_ids in tqdm(loader, desc="predict", leave=False):
        all_imgs = all_imgs.to(device, non_blocking=True)
        group_idx = group_idx.to(device, non_blocking=True)
        view_idx = view_idx.to(device, non_blocking=True)
        B = len(study_ids)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            inst_embs = backbone(all_imgs)
            logits = _unwrap_logits(study_head(inst_embs, group_idx, view_idx, B))
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        all_ids.extend(study_ids)
        all_probs.append(probs)

    probs_all = np.concatenate(all_probs, axis=0).astype(np.float32) if all_probs else np.zeros((0, 0), np.float32)
    return all_ids, probs_all
