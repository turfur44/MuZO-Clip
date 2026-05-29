"""Deterministic param-wise PRNG helpers for MuZO-Clip.

AdaMeZO reconstructs ZO directions by replaying a generator stream.  MuZO-Clip
uses a stricter param-wise seed so the same parameter/block direction can be
recreated without depending on module traversal order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Literal

import torch

NoiseDistribution = Literal["normal", "rademacher"]


def hash64_seed(
    base_seed: int,
    param_name: str,
    param_shape: tuple[int, ...],
    block_index: int = 0,
) -> int:
    """Return a stable 63-bit seed for one parameter block."""

    payload = "|".join(
        [
            str(int(base_seed)),
            param_name,
            ",".join(str(int(dim)) for dim in param_shape),
            str(int(block_index)),
        ]
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def make_zo_noise_like(
    tensor: torch.Tensor,
    base_seed: int,
    param_name: str,
    block_index: int = 0,
    distribution: NoiseDistribution = "normal",
) -> torch.Tensor:
    """Generate a deterministic ZO noise tensor matching ``tensor``.

    The same tuple ``(base_seed, param_name, tensor.shape, block_index,
    distribution)`` always yields the same tensor on the same device/backend.
    This function is used for plus perturbation, minus perturbation, restore,
    and historical momentum reconstruction.
    """

    if not tensor.is_floating_point():
        raise TypeError("MuZO-Clip noise can only be generated for floating tensors")
    if distribution not in ("normal", "rademacher"):
        raise ValueError(f"Unsupported noise distribution: {distribution}")

    seed = hash64_seed(int(base_seed), param_name, tuple(tensor.shape), int(block_index))
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)

    if distribution == "normal":
        noise = torch.randn(
            tensor.shape,
            generator=generator,
            device=tensor.device,
            dtype=torch.float32,
        )
    else:
        noise = torch.randint(
            low=0,
            high=2,
            size=tensor.shape,
            generator=generator,
            device=tensor.device,
            dtype=torch.int8,
        ).to(torch.float32)
        noise.mul_(2.0).sub_(1.0)

    return noise.to(dtype=tensor.dtype)


def iter_param_blocks(
    param: torch.Tensor,
    block_rows: int | None,
) -> Iterator[tuple[int, slice, torch.Tensor]]:
    """Yield deterministic row blocks for a 2D parameter.

    If ``block_rows`` is ``None``, the whole matrix is one block.  Otherwise the
    function yields row chunks and a monotonically increasing block index that
    must be fed back into ``make_zo_noise_like``.
    """

    if param.ndim != 2:
        raise ValueError("MuZO-Clip block iteration only supports 2D tensors")
    if block_rows is None:
        yield 0, slice(None), param
        return
    if block_rows <= 0:
        raise ValueError("block_rows must be positive or None")

    rows = int(param.shape[0])
    for block_index, start in enumerate(range(0, rows, int(block_rows))):
        row_slice = slice(start, min(start + int(block_rows), rows))
        yield block_index, row_slice, param[row_slice]
