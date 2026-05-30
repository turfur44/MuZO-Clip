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


@torch.no_grad()
def batched_zeropower_via_newtonschulz5(G_batch: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched Newton-Schulz orthogonalization for row-block MuZO updates.

    ``G_batch`` must have shape ``[B, M, N]``.  The 2D implementation above is
    intentionally left unchanged; this function mirrors its coefficients and
    per-item normalization while using ``torch.bmm`` for the matrix products.
    """

    if G_batch.ndim != 3:
        raise ValueError("batched_zeropower_via_newtonschulz5 accepts [B, M, N] tensors only")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    out_dtype = G_batch.dtype if G_batch.is_floating_point() else torch.float32
    zeros = torch.zeros_like(G_batch, dtype=out_dtype)
    if G_batch.numel() == 0:
        return zeros

    finite = torch.isfinite(G_batch).flatten(1).all(dim=1)
    norms = G_batch.float().flatten(1).norm(dim=1)
    valid = finite & torch.isfinite(norms) & (norms > 0)

    a, b, c = (3.4445, -4.7750, 2.0315)
    compute_dtype = torch.bfloat16 if G_batch.device.type == "cuda" else torch.float32
    X = G_batch.to(dtype=compute_dtype)

    transposed = G_batch.size(1) > G_batch.size(2)
    if transposed:
        X = X.transpose(1, 2)

    x_norms = X.float().flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1)
    X = X / x_norms

    for _ in range(int(steps)):
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)

    if transposed:
        X = X.transpose(1, 2)

    output_finite = torch.isfinite(X).flatten(1).all(dim=1)
    valid = valid & output_finite
    X = X.to(dtype=out_dtype)
    return torch.where(valid.view(-1, 1, 1), X, zeros)
