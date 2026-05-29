"""Experimental fast-path scaffolding.

The default implementation remains the reference PyTorch path. Fused kernels
are deliberately opt-in and currently unavailable until implemented/tested.
"""

from __future__ import annotations

from typing import Literal

import torch

FastPathBackend = Literal["torch", "fused_rademacher"]


def require_supported_backend(backend: FastPathBackend, distribution: str) -> None:
    if backend == "torch":
        return
    if backend == "fused_rademacher" and distribution != "rademacher":
        raise ValueError("fused_rademacher backend requires distribution='rademacher'")
    if backend == "fused_rademacher":
        raise NotImplementedError(
            "fused_rademacher backend is scaffolded but not implemented yet. "
            "Use backend='torch' for the reference deterministic path."
        )
    raise ValueError(f"Unsupported fast-path backend: {backend}")


def fused_perturb_inplace_rademacher(
    param_block: torch.Tensor,
    *,
    base_seed: int,
    param_hash: int,
    block_index: int,
    scale: float,
) -> None:
    raise NotImplementedError("Triton/CUDA fused perturb kernel is future work")


def fused_momentum_reconstruct_rademacher(
    out_m: torch.Tensor,
    *,
    seeds: torch.Tensor,
    coeffs: torch.Tensor,
    param_hash: int,
    block_index: int,
    normalize: bool,
) -> None:
    raise NotImplementedError("Triton/CUDA fused momentum reconstruction kernel is future work")

