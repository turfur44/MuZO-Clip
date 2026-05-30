#!/usr/bin/env python
"""Microbenchmark MuZO-Clip fused_rademacher kernels.

Correctness is checked before timing. Any mismatch raises and exits non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from muzo_clip.fastpath import (
    fused_momentum_reconstruct_rademacher,
    fused_perturb_inplace_rademacher,
    param_hash64,
    rademacher_counter_reference,
    require_supported_backend,
)

SHAPES = [(1024, 1024), (4096, 4096), (4096, 11008), (11008, 4096)]
HISTORY_LENGTHS = [1, 4, 8]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
MASK32 = (1 << 32) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused_rademacher against torch counter reference")
    parser.add_argument("--jsonl", default="benchmark_fastpath.jsonl")
    parser.add_argument("--csv", default="benchmark_fastpath.csv")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip_bf16", action="store_true")
    parser.add_argument("--max_shape_elements", type=int, default=0, help="0 means run all configured shapes")
    return parser.parse_args()


def cuda_time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


def counter_rademacher_torch(
    shape: tuple[int, int],
    *,
    base_seed: int,
    param_hash: int,
    block_index: int,
    device: torch.device,
) -> torch.Tensor:
    seed_lo = int(base_seed) & MASK32
    seed_hi = (int(base_seed) >> 32) & MASK32
    hash_lo = int(param_hash) & MASK32
    hash_hi = (int(param_hash) >> 32) & MASK32
    offsets = torch.arange(shape[0] * shape[1], device=device, dtype=torch.int64)
    x = offsets & MASK32
    x = torch.bitwise_xor(x, seed_lo)
    x = (x + seed_hi * 747796405) & MASK32
    x = torch.bitwise_xor(x, hash_lo)
    x = (x + hash_hi * 289133645) & MASK32
    x = torch.bitwise_xor(x, (int(block_index) * 668265263) & MASK32)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = (x * 73244475) & MASK32
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 15))
    x = (x * 668265263) & MASK32
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    signs = torch.where((x & 1) == 0, -1.0, 1.0).to(torch.float32)
    return signs.reshape(shape)


def check_counter_reference(shape: tuple[int, int], seed: int, param_hash: int, block_index: int) -> None:
    small_shape = (min(shape[0], 17), min(shape[1], 19))
    torch_ref = counter_rademacher_torch(
        small_shape,
        base_seed=seed,
        param_hash=param_hash,
        block_index=block_index,
        device=torch.device("cuda"),
    )
    python_ref = rademacher_counter_reference(
        small_shape,
        base_seed=seed,
        param_hash=param_hash,
        block_index=block_index,
        device="cuda",
    )
    if not torch.equal(torch_ref, python_ref):
        raise AssertionError(f"torch counter reference mismatch for shape={small_shape}")


def check_perturb(shape: tuple[int, int], dtype: torch.dtype, seed: int, param_hash: int) -> None:
    check_counter_reference(shape, seed, param_hash, 0)
    fused = torch.zeros(shape, device="cuda", dtype=dtype)
    torch_path = torch.zeros_like(fused)
    fused_perturb_inplace_rademacher(fused, base_seed=seed, param_hash=param_hash, block_index=0, scale=1.0)
    torch_path.add_(counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash, block_index=0, device=fused.device).to(dtype))
    torch.cuda.synchronize()
    if not torch.equal(fused, torch_path):
        max_diff = float((fused.float() - torch_path.float()).abs().max().item())
        raise AssertionError(f"perturb mismatch shape={shape} dtype={dtype} max_diff={max_diff}")

    restored = fused.clone()
    fused_perturb_inplace_rademacher(restored, base_seed=seed, param_hash=param_hash, block_index=0, scale=-1.0)
    torch.cuda.synchronize()
    if dtype == torch.float32 and float(restored.abs().max().item()) != 0.0:
        raise AssertionError(f"fp32 perturb restore was not exact for shape={shape}")


def check_reconstruct(shape: tuple[int, int], h: int, param_hash: int) -> tuple[torch.Tensor, torch.Tensor]:
    seeds = torch.tensor([1009 + i * 104729 for i in range(h)], device="cuda", dtype=torch.int64)
    coeffs = torch.tensor([((-1.0) ** i) * (0.5 + 0.125 * i) for i in range(h)], device="cuda", dtype=torch.float32)
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    fused_momentum_reconstruct_rademacher(out, seeds=seeds, coeffs=coeffs, param_hash=param_hash, block_index=0)
    ref = torch.zeros(shape, device="cuda", dtype=torch.float32)
    for seed, coeff in zip(seeds.cpu().tolist(), coeffs.cpu().tolist()):
        ref.add_(
            counter_rademacher_torch(shape, base_seed=int(seed), param_hash=param_hash, block_index=0, device=out.device),
            alpha=float(coeff),
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)
    return seeds, coeffs


def write_rows(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_fastpath requires CUDA")
    require_supported_backend("fused_rademacher", "rademacher")
    dtypes = ["fp32"] if args.skip_bf16 or not torch.cuda.is_bf16_supported() else ["fp32", "bf16"]
    rows: list[dict[str, Any]] = []
    started = time.time()

    for shape in SHAPES:
        if args.max_shape_elements and shape[0] * shape[1] > args.max_shape_elements:
            continue
        param_hash = param_hash64("benchmark.q_proj.weight", shape)
        for dtype_name in dtypes:
            dtype = DTYPES[dtype_name]
            seed = (1 << 63) - 12345
            check_perturb(shape, dtype, seed, param_hash)
            fused_tensor = torch.zeros(shape, device="cuda", dtype=dtype)
            torch_tensor = torch.zeros_like(fused_tensor)

            def fused_perturb():
                fused_perturb_inplace_rademacher(
                    fused_tensor,
                    base_seed=seed,
                    param_hash=param_hash,
                    block_index=0,
                    scale=1e-3,
                )

            def torch_perturb():
                torch_tensor.add_(
                    counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash, block_index=0, device=torch_tensor.device).to(dtype),
                    alpha=1e-3,
                )

            fused_ms = cuda_time_ms(fused_perturb, warmup=args.warmup, iters=args.iters)
            torch_ms = cuda_time_ms(torch_perturb, warmup=args.warmup, iters=args.iters)
            rows.append(
                {
                    "benchmark": "perturb",
                    "shape": f"{shape[0]}x{shape[1]}",
                    "history": None,
                    "dtype": dtype_name,
                    "fused_ms": fused_ms,
                    "torch_ms": torch_ms,
                    "speedup": torch_ms / fused_ms if fused_ms > 0 else None,
                    "elements": shape[0] * shape[1],
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

        for h in HISTORY_LENGTHS:
            seeds, coeffs = check_reconstruct(shape, h, param_hash)
            fused_out = torch.empty(shape, device="cuda", dtype=torch.float32)

            def fused_reconstruct():
                fused_momentum_reconstruct_rademacher(
                    fused_out,
                    seeds=seeds,
                    coeffs=coeffs,
                    param_hash=param_hash,
                    block_index=0,
                )

            def torch_reconstruct():
                ref = torch.zeros(shape, device="cuda", dtype=torch.float32)
                for seed, coeff in zip(seeds.cpu().tolist(), coeffs.cpu().tolist()):
                    ref.add_(
                        counter_rademacher_torch(shape, base_seed=int(seed), param_hash=param_hash, block_index=0, device=ref.device),
                        alpha=float(coeff),
                    )

            fused_ms = cuda_time_ms(fused_reconstruct, warmup=args.warmup, iters=args.iters)
            torch_ms = cuda_time_ms(torch_reconstruct, warmup=args.warmup, iters=args.iters)
            rows.append(
                {
                    "benchmark": "reconstruct",
                    "shape": f"{shape[0]}x{shape[1]}",
                    "history": h,
                    "dtype": "fp32",
                    "fused_ms": fused_ms,
                    "torch_ms": torch_ms,
                    "speedup": torch_ms / fused_ms if fused_ms > 0 else None,
                    "elements": shape[0] * shape[1],
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    for row in rows:
        row["elapsed_wall_sec"] = time.time() - started
    write_rows(rows, Path(args.jsonl), Path(args.csv))


if __name__ == "__main__":
    main()
