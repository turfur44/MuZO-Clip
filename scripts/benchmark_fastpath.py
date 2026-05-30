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
BLOCK_ROWS_CASES: list[int | None] = [512, 1024, None]
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


def block_cases(matrix_shape: tuple[int, int], block_rows: int | None) -> list[tuple[int, tuple[int, int]]]:
    if block_rows is None:
        return [(0, matrix_shape)]
    rows, cols = matrix_shape
    cases: list[tuple[int, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for block_index, start in enumerate(range(0, rows, block_rows)):
        block_shape = (min(block_rows, rows - start), cols)
        if block_shape not in seen:
            cases.append((block_index, block_shape))
            seen.add(block_shape)
    return cases


def block_rows_label(block_rows: int | None) -> str:
    return "full" if block_rows is None else str(block_rows)


def correctness_fail(message: str, **case: object) -> None:
    payload = {"error": message, **case}
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    raise SystemExit(1)


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


def check_sign_sensitivity(shape: tuple[int, int], seed: int, param_hash: int, block_index: int) -> None:
    base = counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash, block_index=block_index, device=torch.device("cuda"))
    same = counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash, block_index=block_index, device=torch.device("cuda"))
    if not torch.equal(base, same):
        correctness_fail("same seed/hash/block produced different signs", shape=shape, block_index=block_index)
    changed_seed = counter_rademacher_torch(shape, base_seed=seed + 1, param_hash=param_hash, block_index=block_index, device=torch.device("cuda"))
    changed_hash = counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash + 1, block_index=block_index, device=torch.device("cuda"))
    changed_block = counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash, block_index=block_index + 1, device=torch.device("cuda"))
    if torch.equal(base, changed_seed):
        correctness_fail("different seed did not change signs", shape=shape, block_index=block_index)
    if torch.equal(base, changed_hash):
        correctness_fail("different hash did not change signs", shape=shape, block_index=block_index)
    if torch.equal(base, changed_block):
        correctness_fail("different block did not change signs", shape=shape, block_index=block_index)


def check_perturb(shape: tuple[int, int], dtype: torch.dtype, seed: int, param_hash: int, block_index: int) -> None:
    check_counter_reference(shape, seed, param_hash, block_index)
    check_sign_sensitivity((min(shape[0], 64), min(shape[1], 64)), seed, param_hash, block_index)
    fused = torch.zeros(shape, device="cuda", dtype=dtype)
    torch_path = torch.zeros_like(fused)
    fused_perturb_inplace_rademacher(fused, base_seed=seed, param_hash=param_hash, block_index=block_index, scale=1.0)
    torch_path.add_(counter_rademacher_torch(shape, base_seed=seed, param_hash=param_hash, block_index=block_index, device=fused.device).to(dtype))
    torch.cuda.synchronize()
    if not torch.equal(fused, torch_path):
        max_diff = float((fused.float() - torch_path.float()).abs().max().item())
        correctness_fail("perturb mismatch", shape=shape, dtype=str(dtype), block_index=block_index, max_diff=max_diff)

    restored = fused.clone()
    fused_perturb_inplace_rademacher(restored, base_seed=seed, param_hash=param_hash, block_index=block_index, scale=-1.0)
    torch.cuda.synchronize()
    if dtype == torch.float32 and float(restored.abs().max().item()) != 0.0:
        correctness_fail("fp32 perturb restore was not exact", shape=shape, block_index=block_index)


def check_reconstruct(shape: tuple[int, int], h: int, param_hash: int, block_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    seeds = torch.tensor([1009 + i * 104729 for i in range(h)], device="cuda", dtype=torch.int64)
    coeffs = torch.tensor([((-1.0) ** i) * (0.5 + 0.125 * i) for i in range(h)], device="cuda", dtype=torch.float32)
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    fused_momentum_reconstruct_rademacher(out, seeds=seeds, coeffs=coeffs, param_hash=param_hash, block_index=block_index)
    ref = torch.zeros(shape, device="cuda", dtype=torch.float32)
    for seed, coeff in zip(seeds.cpu().tolist(), coeffs.cpu().tolist()):
        ref.add_(
            counter_rademacher_torch(shape, base_seed=int(seed), param_hash=param_hash, block_index=block_index, device=out.device),
            alpha=float(coeff),
        )
    torch.cuda.synchronize()
    try:
        torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)
    except AssertionError as exc:
        correctness_fail("reconstruct mismatch", shape=shape, history=h, block_index=block_index, detail=str(exc))
    zero_coeffs = coeffs.clone()
    if h > 1:
        zero_coeffs[1::2] = 0.0
        fused_momentum_reconstruct_rademacher(out, seeds=seeds, coeffs=zero_coeffs, param_hash=param_hash, block_index=block_index)
        ref.zero_()
        for seed, coeff in zip(seeds.cpu().tolist(), zero_coeffs.cpu().tolist()):
            ref.add_(
                counter_rademacher_torch(shape, base_seed=int(seed), param_hash=param_hash, block_index=block_index, device=out.device),
                alpha=float(coeff),
            )
        torch.cuda.synchronize()
        try:
            torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)
        except AssertionError as exc:
            correctness_fail("reconstruct zero-coeff mismatch", shape=shape, history=h, block_index=block_index, detail=str(exc))
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

    for matrix_shape in SHAPES:
        if args.max_shape_elements and matrix_shape[0] * matrix_shape[1] > args.max_shape_elements:
            continue
        param_hash = param_hash64("benchmark.q_proj.weight", matrix_shape)
        for block_rows in BLOCK_ROWS_CASES:
            for block_index, shape in block_cases(matrix_shape, block_rows):
                for dtype_name in dtypes:
                    dtype = DTYPES[dtype_name]
                    seed = (1 << 63) - 12345
                    check_perturb(shape, dtype, seed, param_hash, block_index)
                    fused_tensor = torch.zeros(shape, device="cuda", dtype=dtype)
                    torch_tensor = torch.zeros_like(fused_tensor)

                    def fused_perturb():
                        fused_perturb_inplace_rademacher(
                            fused_tensor,
                            base_seed=seed,
                            param_hash=param_hash,
                            block_index=block_index,
                            scale=1e-3,
                        )

                    def torch_perturb():
                        torch_tensor.add_(
                            counter_rademacher_torch(
                                shape,
                                base_seed=seed,
                                param_hash=param_hash,
                                block_index=block_index,
                                device=torch_tensor.device,
                            ).to(dtype),
                            alpha=1e-3,
                        )

                    fused_ms = cuda_time_ms(fused_perturb, warmup=args.warmup, iters=args.iters)
                    torch_ms = cuda_time_ms(torch_perturb, warmup=args.warmup, iters=args.iters)
                    rows.append(
                        {
                            "benchmark": "perturb",
                            "matrix_shape": f"{matrix_shape[0]}x{matrix_shape[1]}",
                            "block_rows": block_rows_label(block_rows),
                            "block_index": block_index,
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
                    seeds, coeffs = check_reconstruct(shape, h, param_hash, block_index)
                    fused_out = torch.empty(shape, device="cuda", dtype=torch.float32)

                    def fused_reconstruct():
                        fused_momentum_reconstruct_rademacher(
                            fused_out,
                            seeds=seeds,
                            coeffs=coeffs,
                            param_hash=param_hash,
                            block_index=block_index,
                        )

                    def torch_reconstruct():
                        ref = torch.zeros(shape, device="cuda", dtype=torch.float32)
                        for seed, coeff in zip(seeds.cpu().tolist(), coeffs.cpu().tolist()):
                            ref.add_(
                                counter_rademacher_torch(
                                    shape,
                                    base_seed=int(seed),
                                    param_hash=param_hash,
                                    block_index=block_index,
                                    device=ref.device,
                                ),
                                alpha=float(coeff),
                            )

                    fused_ms = cuda_time_ms(fused_reconstruct, warmup=args.warmup, iters=args.iters)
                    torch_ms = cuda_time_ms(torch_reconstruct, warmup=args.warmup, iters=args.iters)
                    rows.append(
                        {
                            "benchmark": "reconstruct",
                            "matrix_shape": f"{matrix_shape[0]}x{matrix_shape[1]}",
                            "block_rows": block_rows_label(block_rows),
                            "block_index": block_index,
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
