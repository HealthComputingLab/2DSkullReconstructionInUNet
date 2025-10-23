"""Lowercase compatibility wrapper for Super_Resolution library.
Re-exports public names from `Library.py`.
"""
from .Library import *

__all__ = [name for name in dir() if not name.startswith("_")]
"""Lowercase compatibility shim for Super_Resolution.

Re-exports the public utilities from `Library.py` so code can import
`Unet_Architecture.Super_Resolution.library` regardless of legacy casing.
"""
from .Library import *

__all__ = getattr(__import__(__name__), "__all__", None) or [
    "SEED",
    "BATCH_SIZE",
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "dice_coefficient",
]
"""Lowercase wrapper to expose legacy Library.py functionality."""
try:
    from .Library import *
except Exception:
    # very small fallback
    def dice_coefficient(*args, **kwargs):
        raise RuntimeError("Legacy Library.py not available")

__all__ = [name for name in globals().keys() if not name.startswith("_")]
"""Lowercase library shim for Super_Resolution.

Re-exports from `Library.py` if present, otherwise provides minimal fallbacks.
"""
try:
    from .Library import *
except Exception:
    import numpy as np

    def dice_coefficient(*args, **kwargs):
        raise RuntimeError("Legacy Library.py missing; install the project's original files")

__all__ = ["dice_coefficient"]
"""compatibility wrapper for Super_Resolution Library
"""
try:
    from .Library import *  # re-export legacy implementation
except Exception:
    import numpy as np

    def dice_coefficient(*a, **k):
        raise RuntimeError("dice_coefficient requires the original Library.py to be present")

__all__ = [n for n in globals().keys() if not n.startswith("_")]
"""Minimal library for Super_Resolution.

If a legacy `Library.py` exists in this package it will be re-exported. If not,
provide a small dice_coefficient implementation compatible with the image
painting helper.
"""

import os

try:
    # Prefer re-exporting a legacy Library.py if present
    from .Library import *  # type: ignore
except Exception:
    import numpy as np

    def dice_coefficient(pred, target, threshold=0.5, eps=1e-8):
        pred = np.asarray(pred)
        target = np.asarray(target)
        if pred.ndim == 2:
            pred = pred[np.newaxis, np.newaxis, ...]
        if target.ndim == 2:
            target = target[np.newaxis, np.newaxis, ...]
        pred_bin = (pred > threshold).astype(np.float32)
        target_bin = (target > threshold).astype(np.float32)
        intersection = (pred_bin * target_bin).sum(axis=(1, 2, 3))
        union = pred_bin.sum(axis=(1, 2, 3)) + target_bin.sum(axis=(1, 2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        return float(dice.mean())

__all__ = ["dice_coefficient"]
"""Lightweight helpers for Super_Resolution.

Provides small utilities that do not import heavy packages at module import time.
Functions that need torch or other ML packages import them lazily.
"""
from typing import Any
import numpy as np
import config as cf

# Basic constants
SEED = 1
BATCH_SIZE = cf.BATCH_SIZE
IMG_HEIGHT = cf.IMAGE_HEIGHT
IMG_WIDTH = cf.IMAGE_WIDTH


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch

        return torch.is_tensor(x)
    except Exception:
        return False


def dice_coefficient(pred: Any, target: Any, threshold: float = 0.5, eps: float = 1e-8) -> float:
    """Dice coefficient working with numpy or torch tensors. Returns a Python float."""
    if _is_torch_tensor(pred) or _is_torch_tensor(target):
        import torch

        pred = pred.clone()
        target = target.clone()
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()
        intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
        union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        return float(dice.mean().item())

    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.ndim == 2:
        pred = pred[np.newaxis, np.newaxis, ...]
    if target.ndim == 2:
        target = target[np.newaxis, np.newaxis, ...]
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > threshold).astype(np.float32)
    intersection = (pred_bin * target_bin).sum(axis=(1, 2, 3))
    union = pred_bin.sum(axis=(1, 2, 3)) + target_bin.sum(axis=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return float(dice.mean())


__all__ = ["SEED", "BATCH_SIZE", "IMG_HEIGHT", "IMG_WIDTH", "dice_coefficient"]

