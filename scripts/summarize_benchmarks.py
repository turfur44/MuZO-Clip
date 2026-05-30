#!/usr/bin/env python
"""Summarize JSONL train benchmark logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MuZO-Clip benchmark JSONL logs")
    parser.add_argument("logs", nargs="+", help="JSONL train.log files or directories containing train.log")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--warmup_rows", type=int, default=3)
    return parser.parse_args()


def iter_log_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def resolve_logs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*/train.log")))
            if (path / "train.log").exists():
                paths.append(path / "train.log")
        else:
            paths.append(path)
    return paths


def summarize_one(path: Path, warmup_rows: int) -> dict[str, Any]:
    rows = list(iter_log_rows(path))
    if not rows:
        raise ValueError(f"No JSON rows found in {path}")
    metric_rows = rows[max(0, int(warmup_rows)) :]
    if not metric_rows:
        metric_rows = rows
    variant = rows[-1].get("variant_name") or rows[-1].get("fast_path_backend") or path.parent.name
    tokens = [float(row["tokens_per_second"]) for row in metric_rows if row.get("tokens_per_second") is not None]
    steps = [float(row["steps_per_second"]) for row in metric_rows if row.get("steps_per_second") is not None]
    forward = [float(row["forward_time_ms"]) for row in metric_rows if row.get("forward_time_ms") is not None]
    update = [float(row["update_time_ms"]) for row in metric_rows if row.get("update_time_ms") is not None]
    peaks = [int(row["gpu_memory_peak"]) for row in rows if row.get("gpu_memory_peak") is not None]
    projection_skipped = sum(1 for row in rows if row.get("projection_skipped"))
    step_skipped = sum(1 for row in rows if row.get("step_skipped"))
    final = rows[-1]
    return {
        "log": str(path),
        "variant": variant,
        "rows": len(rows),
        "warmup_rows": min(max(0, int(warmup_rows)), len(rows)),
        "metric_rows": len(metric_rows),
        "tokens_per_second_mean": mean(tokens) if tokens else None,
        "tokens_per_second_last": tokens[-1] if tokens else None,
        "steps_per_second_mean": mean(steps) if steps else None,
        "steps_per_second_last": steps[-1] if steps else None,
        "forward_time_ms_mean": mean(forward) if forward else None,
        "update_time_ms_mean": mean(update) if update else None,
        "gpu_memory_peak_max": max(peaks) if peaks else None,
        "final_loss_mean": final.get("loss_mean"),
        "final_p_used": final.get("p_used"),
        "final_update_ratio_max": final.get("update_ratio_max"),
        "projection_skipped_count": projection_skipped,
        "step_skipped_count": step_skipped,
        "skipped_steps": projection_skipped + step_skipped,
        "final_step": final.get("step"),
        "distribution": final.get("distribution"),
        "fast_path_backend": final.get("fast_path_backend"),
        "block_rows": final.get("block_rows"),
        "sparse_update_mode": final.get("sparse_update_mode"),
        "sparse_update_groups": final.get("sparse_update_groups"),
    }


def main() -> None:
    args = parse_args()
    summaries = [summarize_one(path, args.warmup_rows) for path in resolve_logs(args.logs)]
    for row in summaries:
        print(json.dumps(row, sort_keys=True), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv:
        fields = sorted({key for row in summaries for key in row})
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
