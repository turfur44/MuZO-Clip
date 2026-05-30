#!/usr/bin/env python
"""Standalone MuZO-Clip training script.

Defaults preserve the reference MuZO-Clip behavior. Profiling, sparse schedules,
token cache loading, and fused-kernel scaffolding are all opt-in.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from muzo_clip.block_config import BlockRows
from muzo_clip.data import iter_batches
from muzo_clip.fastpath import FastPathBackend
from muzo_clip.muzo_optimizer import MuZOClipOptimizer
from muzo_clip.profiling import PhaseProfiler, maybe_torch_profiler
from muzo_clip.sparse_schedule import SparseUpdateMode
from muzo_clip.token_cache import build_token_cache, is_valid_cache, iter_cached_batches, resolve_token_cache


def dtype_from_arg(value: str):
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[value]


def parse_block_rows(value: str) -> BlockRows:
    lowered = value.lower()
    if lowered in ("none", "full"):
        return None
    if lowered == "auto":
        return "auto"
    if lowered == "auto_full":
        return "auto_full"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("block_rows must be positive, none, full, auto, or auto_full")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal LM with MuZO-Clip")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--min_history", type=int, default=4)
    parser.add_argument("--beta_momentum", type=float, default=0.9)
    parser.add_argument("--ns_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--p_clip_value", type=float, default=3.0)
    parser.add_argument("--p_ema_beta", type=float, default=0.95)
    parser.add_argument("--update_ratio_clip", type=float, default=0.01)
    parser.add_argument("--muon_scale", type=float, default=0.2)
    parser.add_argument("--block_rows", type=parse_block_rows, default=256)
    parser.add_argument("--full_block_max_elements", type=int, default=8_388_608)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16", "auto"], default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--loss_mode", choices=["assistant_only", "full_text"], default="assistant_only")
    parser.add_argument("--distribution", choices=["normal", "rademacher"], default="normal")
    parser.add_argument("--fast_path_backend", choices=["torch", "fused_rademacher"], default="torch")
    parser.add_argument("--update_fast_path", choices=["torch", "gpu_stats"], default="torch")
    parser.add_argument("--matrix_update_mode", choices=["block_loop", "batched_blocks"], default="block_loop")
    parser.add_argument("--sparse_update_mode", choices=["off", "round_robin"], default="off")
    parser.add_argument("--sparse_update_groups", type=int, default=1)
    parser.add_argument("--qk_clip_mode", choices=["none", "periodic_eager"], default="none")
    parser.add_argument("--qk_clip_tau", type=float, default=100.0)
    parser.add_argument("--qk_clip_alpha", type=float, default=0.5)
    parser.add_argument("--qk_check_every", type=int, default=50)
    parser.add_argument("--token_cache_mode", choices=["auto", "build", "require", "off"], default="auto")
    parser.add_argument("--token_cache_root", default="token_cache")
    parser.add_argument("--token_cache_overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--parquet_batch_rows", type=int, default=512)
    parser.add_argument("--profile_phases", action="store_true")
    parser.add_argument("--profile_phase_csv", default=None)
    parser.add_argument("--profile_nvtx", action="store_true")
    parser.add_argument("--torch_profile", action="store_true")
    parser.add_argument("--torch_profile_dir", default="profiler_traces")
    parser.add_argument("--torch_profile_wait", type=int, default=5)
    parser.add_argument("--torch_profile_warmup", type=int, default=2)
    parser.add_argument("--torch_profile_active", type=int, default=5)
    parser.add_argument("--torch_profile_repeat", type=int, default=1)
    parser.add_argument("--gc_every", type=int, default=0)
    parser.add_argument("--no-save_final", dest="save_final", action="store_false", default=True)
    args = parser.parse_args()
    if args.seq_len <= 0 or args.batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    if args.lr <= 0 or args.zo_eps <= 0:
        raise ValueError("lr and zo_eps must be positive")
    if args.min_history > args.horizon:
        raise ValueError("min_history must be less than or equal to horizon")
    if args.sparse_update_groups <= 0:
        raise ValueError("sparse_update_groups must be positive")
    if args.full_block_max_elements <= 0:
        raise ValueError("full_block_max_elements must be positive")
    if args.gc_every < 0:
        raise ValueError("gc_every must be non-negative")
    return args


def prune_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0), ignore_errors=True)


def gpu_memory_stats() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    return int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated())


def qk_disabled_stats(reason: str) -> dict[str, object]:
    return {
        "enabled": False,
        "exact_logits": False,
        "fallback_used": False,
        "disabled_reason": reason,
        "qk_smax_max": 0.0,
        "qk_clip_count": 0,
    }


def make_batch_iter(args: argparse.Namespace, tokenizer, tokenizer_source: str) -> Iterator[dict[str, torch.Tensor]]:
    if args.token_cache_mode == "off":
        logging.info("Token cache disabled; using online tokenization")
        return iter_batches(
            tokenizer,
            Path(args.data_path),
            args.text_column,
            args.seq_len,
            args.batch_size,
            args.epochs,
            args.loss_mode,
            args.parquet_batch_rows,
        )
    cache_dir, expected = resolve_token_cache(
        data_path=Path(args.data_path),
        tokenizer_source=tokenizer_source,
        text_column=args.text_column,
        seq_len=args.seq_len,
        loss_mode=args.loss_mode,
        cache_root=Path(args.token_cache_root),
    )
    valid = is_valid_cache(cache_dir, expected)
    if not valid and args.token_cache_mode == "build":
        cache_dir = build_token_cache(
            tokenizer=tokenizer,
            data_path=Path(args.data_path),
            tokenizer_source=tokenizer_source,
            text_column=args.text_column,
            seq_len=args.seq_len,
            loss_mode=args.loss_mode,
            parquet_batch_rows=args.parquet_batch_rows,
            cache_root=Path(args.token_cache_root),
            overwrite=args.token_cache_overwrite,
        )
        valid = True
    if not valid and args.token_cache_mode == "require":
        raise FileNotFoundError(f"No matching pretokenized cache at {cache_dir}")
    if valid:
        logging.info("Using pretokenized cache at %s", cache_dir)
        return iter_cached_batches(cache_dir, batch_size=args.batch_size, epochs=args.epochs)
    logging.info("No matching token cache at %s; using online tokenization", cache_dir)
    return iter_batches(
        tokenizer,
        Path(args.data_path),
        args.text_column,
        args.seq_len,
        args.batch_size,
        args.epochs,
        args.loss_mode,
        args.parquet_batch_rows,
    )


def should_probe_qk(args: argparse.Namespace, step: int, step_stats: dict[str, object]) -> bool:
    return (
        args.qk_clip_mode == "periodic_eager"
        and not bool(step_stats.get("skipped", False))
        and step % args.qk_check_every == 0
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_source = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_arg(args.dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer.save_pretrained(output_dir)

    phase_profiler = PhaseProfiler(
        enabled=args.profile_phases,
        csv_path=args.profile_phase_csv,
        use_cuda_events=True,
        use_nvtx=args.profile_nvtx,
    )
    if args.profile_phases and torch.cuda.is_available():
        logging.warning(
            "--profile_phases uses CUDA event timing with synchronization at phase boundaries; "
            "tokens_per_second in this mode is diagnostic, not normal training throughput."
        )
    optimizer = MuZOClipOptimizer(
        model,
        lr=args.lr,
        zo_eps=args.zo_eps,
        horizon=args.horizon,
        min_history=args.min_history,
        beta_momentum=args.beta_momentum,
        ns_steps=args.ns_steps,
        weight_decay=args.weight_decay,
        p_clip_value=args.p_clip_value,
        p_ema_beta=args.p_ema_beta,
        update_ratio_clip=args.update_ratio_clip,
        distribution=args.distribution,
        block_rows=args.block_rows,
        seed=args.seed,
        normalize_momentum=True,
        muon_scale=args.muon_scale,
        enable_qk_clip=args.qk_clip_mode != "none",
        qk_capture_clean_forward=False,
        qk_clip_tau=args.qk_clip_tau,
        qk_clip_alpha=args.qk_clip_alpha,
        phase_profiler=phase_profiler,
        fast_path_backend=args.fast_path_backend,  # type: ignore[arg-type]
        update_fast_path=args.update_fast_path,  # type: ignore[arg-type]
        matrix_update_mode=args.matrix_update_mode,  # type: ignore[arg-type]
        sparse_update_mode=args.sparse_update_mode,  # type: ignore[arg-type]
        sparse_update_groups=args.sparse_update_groups,
        full_block_max_elements=args.full_block_max_elements,
    )
    if not optimizer.selected_parameter_names():
        raise RuntimeError("MuZO-Clip selected no parameters")

    logging.info("Selected %d MuZO parameter tensors", len(optimizer.selected_parameter_names()))
    logging.info("QK-Clip mode: %s", args.qk_clip_mode)
    logging.info("Token cache mode: %s", args.token_cache_mode)
    logging.info("Block rows: %s", args.block_rows)
    logging.info("Update fast path: %s", args.update_fast_path)
    logging.info("Matrix update mode: %s", args.matrix_update_mode)
    logging.info("Sparse update: %s groups=%d", args.sparse_update_mode, args.sparse_update_groups)

    batch_iter = make_batch_iter(args, tokenizer, tokenizer_source)
    step = int(args.start_step)
    tokens_seen = step * args.batch_size * args.seq_len
    supervised_tokens_seen = 0
    last_log_time = time.perf_counter()
    last_log_tokens = tokens_seen
    last_log_step = step

    with maybe_torch_profiler(
        enabled=args.torch_profile,
        trace_dir=args.torch_profile_dir,
        wait=args.torch_profile_wait,
        warmup=args.torch_profile_warmup,
        active=args.torch_profile_active,
        repeat=args.torch_profile_repeat,
    ) as torch_prof:
        while args.max_steps is None or step < args.start_step + args.max_steps:
            phase_profiler.reset_step()
            with phase_profiler.phase("data_load"):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
            step += 1
            supervised_tokens = int((batch["labels"] != -100).sum().item())
            if supervised_tokens <= 0:
                continue
            with phase_profiler.phase("batch_to_gpu"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

            def loss_closure() -> torch.Tensor:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                return outputs.loss

            proj_stats = optimizer.estimate_projection(loss_closure)
            step_stats = optimizer.step()
            if should_probe_qk(args, step, step_stats):
                qk_stats = optimizer.probe_and_apply_qk_clip(loss_closure, force_eager=True)
            else:
                qk_stats = qk_disabled_stats("QK-Clip disabled by qk_clip_mode=none")
                if args.qk_clip_mode == "periodic_eager":
                    qk_stats["disabled_reason"] = "QK-Clip waiting for qk_check_every interval"

            tokens_seen += int(input_ids.numel())
            supervised_tokens_seen += supervised_tokens
            if torch.cuda.is_available() and (args.profile_phases or args.torch_profile):
                torch.cuda.synchronize()
            mem_alloc, mem_peak = gpu_memory_stats()
            phase_summary = phase_profiler.current_summary()
            forward_time_ms = phase_summary.get("forward_plus_cuda_ms", 0.0) + phase_summary.get(
                "forward_minus_cuda_ms", 0.0
            )
            update_time_ms = (
                phase_summary.get("muzo_reconstruct_cuda_ms", 0.0)
                + phase_summary.get("newton_schulz_cuda_ms", 0.0)
                + phase_summary.get("apply_update_cuda_ms", 0.0)
            )
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            tokens_per_second = (tokens_seen - last_log_tokens) / elapsed
            steps_per_second = (step - last_log_step) / elapsed

            loss_plus = proj_stats.get("loss_plus")
            loss_minus = proj_stats.get("loss_minus")
            loss_mean = 0.5 * (float(loss_plus) + float(loss_minus)) if loss_plus is not None and loss_minus is not None else None
            row = {
                "variant": "muzo_clip",
                "step": step,
                "local_step": step - args.start_step,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "loss_mean": loss_mean,
                "p_raw": proj_stats.get("p_raw"),
                "p_used": proj_stats.get("p_used"),
                "projection_skipped": proj_stats.get("skipped"),
                "projection_skip_reason": proj_stats.get("skip_reason"),
                "step_skipped": step_stats.get("skipped"),
                "step_skip_reason": step_stats.get("skip_reason"),
                "lr": step_stats.get("lr"),
                "zo_eps": args.zo_eps,
                "horizon": args.horizon,
                "min_history": args.min_history,
                "muon_scale": args.muon_scale,
                "block_rows": str(args.block_rows),
                "distribution": args.distribution,
                "fast_path_backend": args.fast_path_backend,
                "update_fast_path": args.update_fast_path,
                "matrix_update_mode": args.matrix_update_mode,
                "full_block_max_elements": args.full_block_max_elements,
                "sparse_update_mode": args.sparse_update_mode,
                "sparse_update_groups": args.sparse_update_groups,
                "active_param_count": step_stats.get("active_param_count"),
                "active_param_names_hash": step_stats.get("active_param_names_hash"),
                "update_rms_mean": step_stats.get("update_rms_mean"),
                "update_ratio_max": step_stats.get("update_ratio_max"),
                "updated_param_count": step_stats.get("updated_param_count"),
                "forward_time_ms": forward_time_ms,
                "update_time_ms": update_time_ms,
                "tokens_per_second": tokens_per_second,
                "steps_per_second": steps_per_second,
                "tokens_seen": tokens_seen,
                "supervised_tokens": supervised_tokens,
                "supervised_tokens_seen": supervised_tokens_seen,
                "gpu_memory_allocated": mem_alloc,
                "gpu_memory_peak": mem_peak,
                "attn_implementation": args.attn_implementation,
                "loss_mode": args.loss_mode,
                "token_cache_mode": args.token_cache_mode,
                "qk_clip_mode": args.qk_clip_mode,
                "qk_smax_max": qk_stats.get("qk_smax_max", 0.0),
                "qk_clip_count": qk_stats.get("qk_clip_count", 0),
                "qk_exact_logits": qk_stats.get("exact_logits", False),
                "qk_fallback_used": qk_stats.get("fallback_used", False),
                "qk_disabled_reason": qk_stats.get("disabled_reason"),
            }
            if step == args.start_step + 1 or step % args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
                last_log_time = now
                last_log_tokens = tokens_seen
                last_log_step = step
            phase_profiler.write_step(
                step,
                {
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": tokens_per_second,
                    "steps_per_second": steps_per_second,
                    "gpu_memory_peak": mem_peak,
                },
            )
            if args.save_every > 0 and step % args.save_every == 0:
                with phase_profiler.phase("logging_checkpoint"):
                    checkpoint_dir = output_dir / f"checkpoint-{step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir, safe_serialization=True)
                    tokenizer.save_pretrained(checkpoint_dir)
                    prune_checkpoints(output_dir, args.save_total_limit)
            if torch_prof is not None:
                torch_prof.step()
            del input_ids, attention_mask, labels
            if args.gc_every > 0 and step % args.gc_every == 0:
                gc.collect()

    if args.save_final:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
