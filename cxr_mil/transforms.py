from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


# -------------------------
# Registry
# -------------------------
_TRANSFORMS: Dict[str, Callable[..., Any]] = {}


def register_transform(name: str):
    def deco(fn: Callable[..., Any]):
        if name in _TRANSFORMS:
            raise KeyError(f"Transform already registered: {name}")
        _TRANSFORMS[name] = fn
        return fn
    return deco


@register_transform("resize")
def _t_resize(size=512):
    # allow int or [h,w]
    if isinstance(size, int):
        size = (size, size)
    return T.Resize(tuple(size))


@register_transform("hflip")
def _t_hflip(p=0.5):
    return T.RandomHorizontalFlip(p=float(p))


@register_transform("rotation")
def _t_rotation(degrees=7):
    return T.RandomRotation(degrees=float(degrees))


@register_transform("color_jitter")
def _t_color_jitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
    return T.ColorJitter(
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        hue=float(hue),
    )


@register_transform("to_tensor")
def _t_to_tensor():
    return T.ToTensor()


@register_transform("normalize")
def _t_normalize(mean, std):
    return T.Normalize(mean=list(mean), std=list(std))

@register_transform("xrv_normalize")
def _t_xrv_normalize(scale=1024.0):
    """
    TorchXRayVision-style normalization.

    Assumes input is a float Tensor in [0, 1] after ToTensor().
    Converts to roughly [-scale, +scale] (default scale=1024).
    """
    scale = float(scale)
    return T.Lambda(lambda x: (2.0 * x - 1.0) * scale)


@register_transform("center_crop_frac")
def _t_center_crop_frac(frac=1.0):
    """
    Center crop to a square that is `frac` of the smallest image side.
    frac=1.0 -> largest centered square (e.g. 1500x2000 -> 1500x1500)
    frac=0.95 -> slightly tighter crop
    """
    frac = float(frac)

    def _crop(img):
        # PIL image
        w, h = img.size
        side = int(min(w, h) * frac)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))

    return T.Lambda(_crop)

@register_transform("resize_with_pad")
def _t_resize_with_pad(size=512, fill=0, interpolation="bilinear"):
    """
    Letterbox resize: preserve aspect ratio, pad to (size, size).
    Works on PIL images.
    """
    size = int(size)

    interp_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    pil_interp = interp_map.get(str(interpolation).lower(), Image.Resampling.BILINEAR)

    def _letterbox(img):
        # PIL image (W, H)
        w, h = img.size
        if w == 0 or h == 0:
            return img

        scale = min(size / w, size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img2 = F.resize(img, [new_h, new_w], interpolation=pil_interp)

        pad_w = size - new_w
        pad_h = size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return F.pad(img2, [left, top, right, bottom], fill=int(fill))

    return T.Lambda(_letterbox)

def _build_pipeline(spec: List[Dict[str, Any]]) -> T.Compose:
    """
    spec example:
      - {name: resize, size: 512}
      - {name: hflip, p: 0.5}
    """
    tfs = []
    for item in spec:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"Each transform must be a dict with a 'name'. Got: {item}")

        name = item["name"]
        kwargs = {k: v for k, v in item.items() if k != "name"}

        if name not in _TRANSFORMS:
            raise KeyError(f"Unknown transform '{name}'. Available: {sorted(_TRANSFORMS)}")

        tfs.append(_TRANSFORMS[name](**kwargs))

    return T.Compose(tfs)


def build_transforms(aug_cfg: Dict[str, Any]) -> Tuple[T.Compose, T.Compose]:
    """
    Expects aug_cfg like:
      {
        "train": [...],
        "val": [...]
      }
    If missing/empty, caller can fall back to defaults.
    """
    train_spec = (aug_cfg.get("train") or [])
    val_spec = (aug_cfg.get("val") or [])

    if not train_spec or not val_spec:
        raise ValueError(
            "augmentations.train and augmentations.val must both be provided "
            "when using YAML-driven augments."
        )

    return _build_pipeline(train_spec), _build_pipeline(val_spec)
