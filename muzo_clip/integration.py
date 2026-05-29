"""Small integration helpers for MuZO-Clip."""

from __future__ import annotations

from typing import Any

import torch

from .muzo_optimizer import MuZOClipOptimizer


def create_muzo_clip_optimizer(model: torch.nn.Module, **kwargs: Any) -> MuZOClipOptimizer:
    """Create a MuZO-Clip optimizer for a HuggingFace or plain torch module."""

    return MuZOClipOptimizer(model, **kwargs)


def qk_clip_mode(optimizer: MuZOClipOptimizer) -> str:
    """Return a human-readable QK-Clip mode for logging."""

    if optimizer.qk_clip is None:
        return "disabled"
    if not optimizer.qk_clip.installed:
        return "disabled-no-qk-modules"
    return "capture-pre-softmax-logits"
