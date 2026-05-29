"""Newton-Schulz matrix orthogonalization for MuZO-Clip.

Adapted from KellerJordan/Muon ``zeropower_via_newtonschulz5``.
Source reference: external_refs/Muon/muon.py
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the zeroth power / orthogonalized matrix of ``G``.

    The quintic Newton-Schulz coefficients are copied from KellerJordan/Muon.
    This wrapper is intentionally stricter for experimental optimizer safety:
    only 2D matrices are accepted, zero/non-finite inputs return zeros, and CPU
    inputs compute in fp32 instead of forcing bf16 matmul support.
    """

    if G.ndim != 2:
        raise ValueError("zeropower_via_newtonschulz5 accepts 2D tensors only")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    out_dtype = G.dtype if G.is_floating_point() else torch.float32
    zeros = torch.zeros_like(G, dtype=out_dtype)

    finite = torch.isfinite(G).all()
    if not bool(finite.item()):
        logger.warning("Newton-Schulz input is non-finite; returning zeros")
        return zeros

    norm = G.float().norm()
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        logger.warning("Newton-Schulz input has zero or invalid norm; returning zeros")
        return zeros

    a, b, c = (3.4445, -4.7750, 2.0315)
    compute_dtype = torch.bfloat16 if G.device.type == "cuda" else torch.float32
    X = G.to(dtype=compute_dtype)

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT

    X = X / (X.float().norm() + 1e-7)

    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    if not bool(torch.isfinite(X).all().item()):
        logger.warning("Newton-Schulz produced non-finite output; returning zeros")
        return zeros

    return X.to(dtype=out_dtype)
