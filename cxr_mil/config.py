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

    # model
    backbone_name: str = "densenet121"
    backbone_pretrained: bool = True
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)

    # optional: load custom pretrained weights (e.g., CheXpert)
    backbone_checkpoint_path: Optional[str] = None
    backbone_checkpoint_key: Optional[str] = None  # e.g. "state_dict", "model", "backbone"
    backbone_prefix_strip: Optional[str] = None    # e.g. "module.", "model.", "backbone."
    backbone_load_strict: bool = False            # False recommended (ignores head mismatch)
    backbone_adapt_input_conv: str = "auto"      # "auto"|"off": adapt 1ch<->3ch conv weights

    head_name: str = "view_aware_gated_attn"
    head_kwargs: Dict[str, Any] = field(default_factory=dict)

    # cv
    n_splits: int = 5
    select_fold: Optional[int] = None  # 1..n_splits or None for all folds

    # training
    total_epochs: int = 20
    frozen_epochs: int = 2
    head_lr: float = 3e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    accum_steps: int = 4
    study_batch_size: int = 48

    # performance
    num_workers: int = 8

    # augmentations (YAML-driven)
    augmentations: Dict[str, Any] = field(default_factory=dict)
