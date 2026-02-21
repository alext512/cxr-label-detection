from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    # data
    root: str
    csv_name: str = "train1.csv"
    img_subdir: str = "train1"

    # experiment
    run_name: str = "densenet512_attn_posw_cv_v1"
    seed: int = 42
    output_root: str = "runs"

    # model
    backbone_name: str = "densenet121"
    backbone_pretrained: bool = True
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)

    # optional: load custom pretrained weights (e.g., CheXpert)
    backbone_checkpoint_path: Optional[str] = None
    backbone_checkpoint_key: Optional[str] = None
    backbone_prefix_strip: Optional[str] = None
    backbone_load_strict: bool = False
    backbone_adapt_input_conv: str = "auto"

    head_name: str = "view_aware_gated_attn"
    head_kwargs: Dict[str, Any] = field(default_factory=dict)

    # cv
    n_splits: int = 5
    select_fold: Optional[int] = None

    # training
    total_epochs: int = 20
    frozen_epochs: int = 2
    head_lr: float = 3e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    accum_steps: int = 4
    study_batch_size: int = 48
    max_grad_norm: Optional[float] = None

    # loss
    loss_name: str = "weighted_bce"  # weighted_bce | bce | asl
    pos_weight_clip: float = 50.0
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    asl_eps: float = 1e-8

    # scheduler
    scheduler_name: str = "reduce_on_plateau"  # reduce_on_plateau | cosine_warmup
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    warmup_steps: int = 0
    min_lr_ratio: float = 0.05

    # ema
    use_ema: bool = False
    ema_decay: float = 0.999

    # performance
    num_workers: int = 8

    # augmentations (YAML-driven)
    augmentations: Dict[str, Any] = field(default_factory=dict)

    # optional analysis defaults
    analysis: Dict[str, Any] = field(default_factory=dict)
