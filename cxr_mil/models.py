from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models


# -----------------------------
# Optional: custom checkpoint loading
# -----------------------------

def get_backbone_feature_module(backbone: nn.Module) -> nn.Module:
    """Return the module that should be optimized when training the backbone.

    DenseNet has `.features`; many other torchvision models do not. This helper
    keeps the training code generic.
    """
    return getattr(backbone, "features", backbone)


def _strip_prefix_if_present(k: str, prefix: Optional[str]) -> str:
    if prefix and k.startswith(prefix):
        return k[len(prefix):]
    return k


def _maybe_adapt_input_conv(state: Dict[str, torch.Tensor], model: nn.Module, mode: str = "auto") -> None:
    """If first conv expects different in_channels (1 vs 3), adapt weights in-place.

    This supports common cases where medical checkpoints are trained on 1-channel
    images but the pipeline uses RGB (3-channel) tensors, or vice versa.
    """
    if mode != "auto":
        return

    model_sd = model.state_dict()
    # find a likely first conv weight key present in both
    candidate_keys = [k for k in state.keys() if k.endswith("conv1.weight")]
    if not candidate_keys:
        # DenseNet stem is often "features.conv0.weight"
        candidate_keys = [k for k in state.keys() if k.endswith("features.conv0.weight")]
    if not candidate_keys:
        return

    k = candidate_keys[0]
    if k not in model_sd:
        return

    w_ckpt = state[k]
    w_model = model_sd[k]
    if not (hasattr(w_ckpt, "shape") and hasattr(w_model, "shape")):
        return
    if len(w_ckpt.shape) != 4 or len(w_model.shape) != 4:
        return

    in_ckpt = w_ckpt.shape[1]
    in_model = w_model.shape[1]
    if in_ckpt == in_model:
        return

    # 1 -> 3: repeat and scale so activation magnitudes stay similar
    if in_ckpt == 1 and in_model == 3:
        state[k] = w_ckpt.repeat(1, 3, 1, 1) / 3.0
    # 3 -> 1: average RGB filters
    elif in_ckpt == 3 and in_model == 1:
        state[k] = w_ckpt.mean(dim=1, keepdim=True)


def load_backbone_weights(
    backbone: nn.Module,
    checkpoint_path: str,
    *,
    checkpoint_key: Optional[str] = None,
    prefix_strip: Optional[str] = None,
    strict: bool = False,
    adapt_input_conv: str = "auto",
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into `backbone` with best-effort key handling.

    Supports raw state_dict checkpoints or wrapper dicts like:
      - {"state_dict": ...}, {"model": ...}, {"backbone": ...}, etc.

    If strict=False (recommended), only parameters with matching names AND shapes
    are loaded; classifier/head weights are ignored automatically.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    state = ckpt
    if isinstance(ckpt, dict):
        if checkpoint_key is not None and checkpoint_key in ckpt and isinstance(ckpt[checkpoint_key], dict):
            state = ckpt[checkpoint_key]
        else:
            for k in ["state_dict", "model", "backbone", "net", "encoder"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    break

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at: {checkpoint_path}")

    # normalize keys (strip prefix if requested)
    state = {_strip_prefix_if_present(k, prefix_strip): v for k, v in state.items()}

    # optional: adapt first conv in_channels
    _maybe_adapt_input_conv(state, backbone, mode=adapt_input_conv)

    if strict:
        missing, unexpected = backbone.load_state_dict(state, strict=True)
        return {"strict": True, "missing": missing, "unexpected": unexpected, "ckpt_keys": len(state)}

    model_sd = backbone.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    return {
        "strict": False,
        "ckpt_keys": len(state),
        "loaded": len(filtered),
        "filtered_out": skipped,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
    }



# -----------------------------
# Backbone registry / factory
# -----------------------------

BackboneFactory = Callable[..., Tuple[nn.Module, int]]
_BACKBONES: Dict[str, BackboneFactory] = {}


def register_backbone(name: str) -> Callable[[BackboneFactory], BackboneFactory]:
    def deco(fn: BackboneFactory) -> BackboneFactory:
        _BACKBONES[name] = fn
        return fn
    return deco


def list_backbones() -> Tuple[str, ...]:
    return tuple(sorted(_BACKBONES.keys()))


def build_backbone(name: str, pretrained: bool = True, **kwargs: Any) -> Tuple[nn.Module, int]:
    """Build a per-image feature extractor that returns an embedding vector.

    Returns (backbone, emb_dim). The backbone must output a tensor [B, emb_dim]
    when called with images [B, C, H, W].
    """
    if name not in _BACKBONES:
        raise KeyError(f"Unknown backbone '{name}'. Available: {list_backbones()}")
    return _BACKBONES[name](pretrained=pretrained, **kwargs)


@register_backbone("densenet121")
def _densenet121(pretrained: bool = True, **_: Any) -> Tuple[nn.Module, int]:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.densenet121(weights=weights)
    emb_dim = m.classifier.in_features
    m.classifier = nn.Identity()
    return m, emb_dim


@register_backbone("resnet50")
def _resnet50(pretrained: bool = True, **_: Any) -> Tuple[nn.Module, int]:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    emb_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, emb_dim


@register_backbone("efficientnet_b0")
def _efficientnet_b0(pretrained: bool = True, **_: Any) -> Tuple[nn.Module, int]:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    # classifier is Sequential([Dropout, Linear])
    emb_dim = m.classifier[-1].in_features
    m.classifier = nn.Identity()
    return m, emb_dim


# -----------------------------
# Heads registry / factory
# -----------------------------

HeadFactory = Callable[..., nn.Module]
_HEADS: Dict[str, HeadFactory] = {}


def register_head(name: str) -> Callable[[HeadFactory], HeadFactory]:
    def deco(fn: HeadFactory) -> HeadFactory:
        _HEADS[name] = fn
        return fn
    return deco


def list_heads() -> Tuple[str, ...]:
    return tuple(sorted(_HEADS.keys()))


def build_head(name: str, emb_dim: int, num_classes: int, **kwargs: Any) -> nn.Module:
    """Build a study-level head.

    Contract: head.forward(inst_embs, group_idx, view_idx, batch_size) -> (logits, aux)
    - inst_embs: [N, emb_dim] embeddings for all images in the batch
    - group_idx: [N] mapping each instance to a bag (0..B-1)
    - view_idx:  [N] 0=frontal, 1=lateral (can be ignored by non-view-aware heads)
    - batch_size: number of bags in the batch
    """
    if name not in _HEADS:
        raise KeyError(f"Unknown head '{name}'. Available: {list_heads()}")
    return _HEADS[name](emb_dim=emb_dim, num_classes=num_classes, **kwargs)


# -----------------------------
# MIL building blocks
# -----------------------------

class GatedAttnPool(nn.Module):
    """Standard gated-attention pooling for MIL."""

    def __init__(self, emb_dim: int, attn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(emb_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(emb_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: [n, emb_dim]
        a = self.w(self.V(x) * self.U(x)).squeeze(-1)  # [n]
        a = torch.softmax(a, dim=0)
        z = (a.unsqueeze(-1) * x).sum(dim=0)           # [emb_dim]
        return z, a


# -----------------------------
# Heads
# -----------------------------

@register_head("mean_pool")
class MeanPoolStudyHead(nn.Module):
    """Mean over instances per study, then an MLP."""

    def __init__(self, emb_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inst_embs: torch.Tensor, group_idx: torch.Tensor, view_idx: torch.Tensor, batch_size: int):
        logits_list = []
        for b in range(batch_size):
            m = group_idx == b
            emb_b = inst_embs[m]
            if emb_b.numel() == 0:
                z = torch.zeros((inst_embs.shape[1],), device=inst_embs.device, dtype=inst_embs.dtype)
            else:
                z = emb_b.mean(dim=0)
            logits_list.append(self.mlp(z))
        return torch.stack(logits_list, dim=0), None


@register_head("gated_attn")
class GatedAttnStudyHead(nn.Module):
    """Single attention pool over all instances in the study."""

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = GatedAttnPool(emb_dim, attn_dim=attn_dim, dropout=attn_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inst_embs: torch.Tensor, group_idx: torch.Tensor, view_idx: torch.Tensor, batch_size: int):
        logits_list = []
        for b in range(batch_size):
            m = group_idx == b
            emb_b = inst_embs[m]
            if emb_b.numel() == 0:
                z = torch.zeros((inst_embs.shape[1],), device=inst_embs.device, dtype=inst_embs.dtype)
            else:
                z, _ = self.pool(emb_b)
            logits_list.append(self.mlp(z))
        return torch.stack(logits_list, dim=0), None


@register_head("view_aware_gated_attn")
class ViewAwareAttnStudyHead(nn.Module):
    """Separate attention pooling for frontal and lateral instances, then fuse."""

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.pool_front = GatedAttnPool(emb_dim, attn_dim=attn_dim, dropout=attn_dropout)
        self.pool_lat = GatedAttnPool(emb_dim, attn_dim=attn_dim, dropout=attn_dropout)

        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inst_embs: torch.Tensor, group_idx: torch.Tensor, view_idx: torch.Tensor, batch_size: int):
        logits_list = []
        for b in range(batch_size):
            m = group_idx == b
            emb_b = inst_embs[m]
            views_b = view_idx[m]

            if emb_b.numel() == 0:
                z = torch.zeros((inst_embs.shape[1] * 2,), device=inst_embs.device, dtype=inst_embs.dtype)
                logits_list.append(self.fuse(z))
                continue

            front_mask = views_b == 0
            lat_mask = views_b == 1

            if front_mask.any():
                zf, _ = self.pool_front(emb_b[front_mask])
            else:
                zf = emb_b.mean(dim=0)

            if lat_mask.any():
                zl, _ = self.pool_lat(emb_b[lat_mask])
            else:
                zl = emb_b.mean(dim=0)

            z = torch.cat([zf, zl], dim=-1)
            logits_list.append(self.fuse(z))

        return torch.stack(logits_list, dim=0), None
