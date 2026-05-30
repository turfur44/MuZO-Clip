#!/usr/bin/env python
"""Generate or run MuZO-Clip training benchmark variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

VARIANTS: dict[str, dict[str, str]] = {
    "torch_rademacher_1024": {
        "distribution": "rademacher",
        "fast_path_backend": "torch",
        "block_rows": "1024",
        "sparse_update_mode": "off",
        "sparse_update_groups": "1",
    },
    "fused_rademacher_1024": {
        "distribution": "rademacher",
        "fast_path_backend": "fused_rademacher",
        "block_rows": "1024",
        "sparse_update_mode": "off",
        "sparse_update_groups": "1",
    },
    "fused_rademacher_auto": {
        "distribution": "rademacher",
        "fast_path_backend": "fused_rademacher",
        "block_rows": "auto",
        "sparse_update_mode": "off",
        "sparse_update_groups": "1",
    },
    "torch_normal_256": {
        "distribution": "normal",
        "fast_path_backend": "torch",
        "block_rows": "256",
        "sparse_update_mode": "off",
        "sparse_update_groups": "1",
    },
    "torch_rademacher_256": {
        "distribution": "rademacher",
        "fast_path_backend": "torch",
        "block_rows": "256",
        "sparse_update_mode": "off",
        "sparse_update_groups": "1",
    },
    "fused_rademacher_auto_sparse2": {
        "distribution": "rademacher",
        "fast_path_backend": "fused_rademacher",
        "block_rows": "auto",
        "sparse_update_mode": "round_robin",
        "sparse_update_groups": "2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuZO-Clip train variant benchmarks")
    parser.add_argument("--mode", choices=["dry_run", "run", "profile_one"], default="dry_run")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_root", default="benchmark_runs")
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--loss_mode", default="assistant_only")
    parser.add_argument("--token_cache_mode", default="require")
    parser.add_argument("--token_cache_root", default="token_cache")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="fused_rademacher_auto")
    parser.add_argument("--variants", default=None, help="Comma-separated variant list for dry_run/run modes")
    parser.add_argument("--extra_arg", action="append", default=[], help="Extra argument token appended verbatim")
    return parser.parse_args()


def base_command(args: argparse.Namespace, variant_name: str, config: dict[str, str]) -> list[str]:
    output_dir = Path(args.output_root) / variant_name
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("train_muzo_clip.py")),
        "--model_name",
        args.model_name,
        "--data_path",
        args.data_path,
        "--text_column",
        args.text_column,
        "--output_dir",
        str(output_dir),
        "--seq_len",
        str(args.seq_len),
        "--batch_size",
        str(args.batch_size),
        "--max_steps",
        str(args.max_steps),
        "--dtype",
        args.dtype,
        "--attn_implementation",
        args.attn_implementation,
        "--loss_mode",
        args.loss_mode,
        "--token_cache_mode",
        args.token_cache_mode,
        "--token_cache_root",
        args.token_cache_root,
        "--qk_clip_mode",
        "none",
        "--log_every",
        str(args.log_every),
        "--save_every",
        "0",
        "--no-save_final",
        "--distribution",
        config["distribution"],
        "--fast_path_backend",
        config["fast_path_backend"],
        "--block_rows",
        config["block_rows"],
        "--sparse_update_mode",
        config["sparse_update_mode"],
        "--sparse_update_groups",
        config["sparse_update_groups"],
    ]
    if args.tokenizer_name:
        command.extend(["--tokenizer_name", args.tokenizer_name])
    command.extend(args.extra_arg)
    return command


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise SystemExit(f"variant failed rc={process.returncode}, log={log_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "profile_one":
        selected = {args.variant: VARIANTS[args.variant]}
    elif args.variants:
        names = [item.strip() for item in args.variants.split(",") if item.strip()]
        unknown = [name for name in names if name not in VARIANTS]
        if unknown:
            raise SystemExit(f"Unknown variants: {', '.join(unknown)}")
        selected = {name: VARIANTS[name] for name in names}
    else:
        selected = VARIANTS
    for name, config in selected.items():
        command = base_command(args, name, config)
        output_dir = Path(args.output_root) / name
        if args.mode == "profile_one":
            command.extend(
                [
                    "--max_steps",
                    str(min(args.max_steps, 10)),
                    "--profile_phases",
                    "--profile_phase_csv",
                    str(output_dir / "phase_times.csv"),
                    "--torch_profile",
                    "--torch_profile_dir",
                    str(output_dir / "profiler_traces"),
                ]
            )
        record: dict[str, Any] = {"variant": name, "command": command, "output_dir": str(output_dir)}
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.mode in ("run", "profile_one"):
            run_command(command, output_dir / "train.log")


if __name__ == "__main__":
    main()
