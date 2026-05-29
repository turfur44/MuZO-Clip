#!/usr/bin/env python
"""Build a binary pretokenized dataset cache for MuZO-Clip training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from muzo_clip.token_cache import build_token_cache, read_cache_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize text/parquet data into fixed-length .bin files")
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--loss_mode", choices=["assistant_only", "full_text"], default="assistant_only")
    parser.add_argument("--token_cache_root", default="token_cache")
    parser.add_argument("--parquet_batch_rows", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
    cache_dir = build_token_cache(
        tokenizer=tokenizer,
        data_path=Path(args.data_path),
        tokenizer_source=args.tokenizer_name,
        text_column=args.text_column,
        seq_len=args.seq_len,
        loss_mode=args.loss_mode,
        parquet_batch_rows=args.parquet_batch_rows,
        cache_root=Path(args.token_cache_root),
        overwrite=args.overwrite,
    )
    metadata = read_cache_metadata(cache_dir) or {}
    print(f"cache_dir={cache_dir}")
    print(f"num_sequences={metadata.get('num_sequences')}")
    print(f"total_tokens={metadata.get('total_tokens')}")
    print(f"supervised_tokens={metadata.get('supervised_tokens')}")


if __name__ == "__main__":
    main()

