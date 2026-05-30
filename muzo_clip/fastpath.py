"""Optional fast-path kernels for MuZO-Clip experiments.

The default optimizer path remains pure PyTorch. Triton is imported lazily and
only required when ``fast_path_backend="fused_rademacher"`` is explicitly used.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import torch

FastPathBackend = Literal["torch", "fused_rademacher"]

triton = None
tl = None
_PERTURB_KERNEL = None
_RECONSTRUCT_KERNEL = None
_C_SEED_HI = 747796405
_C_HASH_HI = 289133645
_C_BLOCK = 668265263
_C_MIX_A = 73244475
_C_MIX_B = 668265263


def ensure_triton_available() -> None:
    """Import Triton lazily for opt-in fused kernels."""

    global triton, tl
    if triton is not None and tl is not None:
        return
    try:
        import triton as triton_module
        import triton.language as tl_module
    except ImportError as exc:
        raise ImportError(
            "fast_path_backend='fused_rademacher' requires Triton. "
            "Install MuZO-Clip with the optional fast extra: pip install 'muzo-clip[fast]'."
        ) from exc
    triton = triton_module
    tl = tl_module


def require_supported_backend(backend: FastPathBackend, distribution: str) -> None:
    if backend == "torch":
        return
    if backend == "fused_rademacher" and distribution != "rademacher":
        raise ValueError("fused_rademacher backend requires distribution='rademacher'")
    if backend == "fused_rademacher":
        ensure_triton_available()
        return
    raise ValueError(f"Unsupported fast-path backend: {backend}")


def param_hash64(param_name: str, param_shape: tuple[int, ...]) -> int:
    payload = "|".join([param_name, ",".join(str(int(dim)) for dim in param_shape)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _split_u64(value: int) -> tuple[int, int]:
    value = int(value) & ((1 << 64) - 1)
    return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF


def _require_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for fused_rademacher")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous for fused_rademacher")


def _launch_grid(numel: int) -> tuple[int]:
    assert triton is not None
    return (triton.cdiv(numel, 256),)


def _get_perturb_kernel():
    global _PERTURB_KERNEL
    ensure_triton_available()
    if _PERTURB_KERNEL is not None:
        return _PERTURB_KERNEL

    @triton.jit
    def _kernel(
        ptr,
        n_elements,
        seed_lo,
        seed_hi,
        hash_lo,
        hash_hi,
        block_index,
        scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        zero = offsets * 0
        seed_lo = (zero + seed_lo).to(tl.uint32)
        seed_hi = (zero + seed_hi).to(tl.uint32)
        hash_lo = (zero + hash_lo).to(tl.uint32)
        hash_hi = (zero + hash_hi).to(tl.uint32)
        block_index = (zero + block_index).to(tl.uint32)
        x = offsets.to(tl.uint32)
        x ^= seed_lo
        x += seed_hi * 747796405
        x ^= hash_lo
        x += hash_hi * 289133645
        x ^= block_index * 668265263
        x ^= x >> 16
        x *= 73244475
        x ^= x >> 15
        x *= 668265263
        x ^= x >> 16
        sign = tl.where((x & 1) == 0, -1.0, 1.0)
        values = tl.load(ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(ptr + offsets, values + sign * scale, mask=mask)

    _PERTURB_KERNEL = _kernel
    return _PERTURB_KERNEL


def _get_reconstruct_kernel():
    global _RECONSTRUCT_KERNEL
    ensure_triton_available()
    if _RECONSTRUCT_KERNEL is not None:
        return _RECONSTRUCT_KERNEL

    @triton.jit
    def _kernel(
        out_ptr,
        seeds_ptr,
        coeffs_ptr,
        n_elements,
        n_history: tl.constexpr,
        hash_lo,
        hash_hi,
        block_index,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        zero = offsets * 0
        hash_lo = (zero + hash_lo).to(tl.uint32)
        hash_hi = (zero + hash_hi).to(tl.uint32)
        block_index = (zero + block_index).to(tl.uint32)
        acc = tl.zeros((BLOCK_SIZE,), tl.float32)
        for history_index in range(0, n_history):
            seed = tl.load(seeds_ptr + history_index).to(tl.uint64)
            seed_lo = seed.to(tl.uint32)
            seed_hi = (seed >> 32).to(tl.uint32)
            coeff = tl.load(coeffs_ptr + history_index).to(tl.float32)
            x = offsets.to(tl.uint32)
            x ^= seed_lo
            x += seed_hi * 747796405
            x ^= hash_lo
            x += hash_hi * 289133645
            x ^= block_index * 668265263
            x ^= x >> 16
            x *= 73244475
            x ^= x >> 15
            x *= 668265263
            x ^= x >> 16
            sign = tl.where((x & 1) == 0, -1.0, 1.0)
            acc += coeff * sign
        tl.store(out_ptr + offsets, acc, mask=mask)

    _RECONSTRUCT_KERNEL = _kernel
    return _RECONSTRUCT_KERNEL


def fused_perturb_inplace_rademacher(
    param_block: torch.Tensor,
    *,
    base_seed: int,
    param_hash: int,
    block_index: int,
    scale: float,
) -> None:
    """Add deterministic counter-based Rademacher signs to ``param_block``."""

    ensure_triton_available()
    _require_cuda_contiguous(param_block, "param_block")
    seed_lo, seed_hi = _split_u64(base_seed)
    hash_lo, hash_hi = _split_u64(param_hash)
    kernel = _get_perturb_kernel()
    numel = int(param_block.numel())
    kernel[_launch_grid(numel)](
        param_block,
        numel,
        seed_lo,
        seed_hi,
        hash_lo,
        hash_hi,
        int(block_index),
        float(scale),
        BLOCK_SIZE=256,
    )


def fused_momentum_reconstruct_rademacher(
    out_m: torch.Tensor,
    *,
    seeds: torch.Tensor,
    coeffs: torch.Tensor,
    param_hash: int,
    block_index: int,
) -> None:
    """Write fused counter-Rademacher momentum reconstruction into ``out_m``."""

    ensure_triton_available()
    _require_cuda_contiguous(out_m, "out_m")
    if out_m.dtype != torch.float32:
        raise TypeError("out_m must be float32")
    if seeds.ndim != 1 or coeffs.ndim != 1 or seeds.numel() != coeffs.numel():
        raise ValueError("seeds and coeffs must be 1D tensors with the same length")
    if seeds.numel() == 0:
        out_m.zero_()
        return
    if seeds.device != out_m.device or seeds.dtype != torch.int64:
        seeds = seeds.to(device=out_m.device, dtype=torch.int64, non_blocking=True)
    if coeffs.device != out_m.device or coeffs.dtype != torch.float32:
        coeffs = coeffs.to(device=out_m.device, dtype=torch.float32, non_blocking=True)
    if not seeds.is_contiguous():
        raise RuntimeError("seeds must be contiguous for fused_rademacher")
    if not coeffs.is_contiguous():
        raise RuntimeError("coeffs must be contiguous for fused_rademacher")
    hash_lo, hash_hi = _split_u64(param_hash)
    kernel = _get_reconstruct_kernel()
    numel = int(out_m.numel())
    kernel[_launch_grid(numel)](
        out_m,
        seeds,
        coeffs,
        numel,
        int(seeds.numel()),
        hash_lo,
        hash_hi,
        int(block_index),
        BLOCK_SIZE=256,
    )


def rademacher_counter_reference(
    shape: tuple[int, ...],
    *,
    base_seed: int,
    param_hash: int,
    block_index: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Slow deterministic reference for the fused counter PRNG."""

    seed_lo, seed_hi = _split_u64(base_seed)
    hash_lo, hash_hi = _split_u64(param_hash)
    signs = []
    for offset in range(math_prod(shape)):
        x = offset & 0xFFFFFFFF
        x ^= seed_lo
        x = (x + seed_hi * _C_SEED_HI) & 0xFFFFFFFF
        x ^= hash_lo
        x = (x + hash_hi * _C_HASH_HI) & 0xFFFFFFFF
        x ^= (int(block_index) * _C_BLOCK) & 0xFFFFFFFF
        x ^= x >> 16
        x = (x * _C_MIX_A) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * _C_MIX_B) & 0xFFFFFFFF
        x ^= x >> 16
        signs.append(-1.0 if (x & 1) == 0 else 1.0)
    return torch.tensor(signs, dtype=torch.float32, device=device).reshape(shape)


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result
