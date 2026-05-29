"""Pretokenized fixed-length dataset cache for MuZO-Clip training."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .data import encode_text, iter_texts

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "config.json",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer_source: str) -> str:
    source_path = Path(tokenizer_source)
    digest = hashlib.sha256()
    digest.update(str(tokenizer_source).encode("utf-8"))
    if source_path.exists():
        root = source_path.resolve()
        digest.update(str(root).encode("utf-8"))
        for name in TOKENIZER_FILES:
            path = root / name
            if path.exists() and path.is_file():
                digest.update(name.encode("utf-8"))
                digest.update(str(path.stat().st_size).encode("ascii"))
                digest.update(_hash_file(path).encode("ascii"))
    return digest.hexdigest()


def expected_metadata(
    *,
    data_path: Path,
    tokenizer_source: str,
    text_column: str,
    seq_len: int,
    loss_mode: str,
) -> dict[str, object]:
    resolved_data = data_path.resolve()
    stat = resolved_data.stat()
    tokenizer_hash = tokenizer_fingerprint(tokenizer_source)
    identity = {
        "version": CACHE_VERSION,
        "format": "muzo_pretok_bin",
        "data_path": str(resolved_data),
        "data_size": int(stat.st_size),
        "data_mtime_ns": int(stat.st_mtime_ns),
        "tokenizer_source": str(tokenizer_source),
        "tokenizer_fingerprint": tokenizer_hash,
        "text_column": text_column,
        "seq_len": int(seq_len),
        "loss_mode": loss_mode,
        "dtype": "int32",
    }
    key_material = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity["cache_key"] = hashlib.sha256(key_material).hexdigest()[:24]
    return identity


def cache_path_for(root: Path, metadata: dict[str, object]) -> Path:
    data_stem = Path(str(metadata["data_path"])).stem
    return root / f"{data_stem}_{metadata['cache_key']}"


def read_cache_metadata(cache_dir: Path) -> dict[str, object] | None:
    path = cache_dir / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def metadata_matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    keys = (
        "version",
        "format",
        "data_path",
        "data_size",
        "data_mtime_ns",
        "tokenizer_fingerprint",
        "text_column",
        "seq_len",
        "loss_mode",
        "dtype",
        "cache_key",
    )
    return all(actual.get(key) == expected.get(key) for key in keys)


def is_valid_cache(cache_dir: Path, expected: dict[str, object]) -> bool:
    actual = read_cache_metadata(cache_dir)
    if actual is None or not metadata_matches(actual, expected):
        return False
    num_sequences = int(actual.get("num_sequences", 0))
    seq_len = int(actual.get("seq_len", 0))
    if num_sequences <= 0 or seq_len <= 0:
        return False
    expected_bytes = num_sequences * seq_len * np.dtype(np.int32).itemsize
    try:
        return (cache_dir / "input_ids.bin").stat().st_size == expected_bytes and (
            cache_dir / "labels.bin"
        ).stat().st_size == expected_bytes
    except FileNotFoundError:
        return False


def build_token_cache(
    *,
    tokenizer,
    data_path: Path,
    tokenizer_source: str,
    text_column: str,
    seq_len: int,
    loss_mode: str,
    parquet_batch_rows: int,
    cache_root: Path,
    overwrite: bool = False,
) -> Path:
    expected = expected_metadata(
        data_path=data_path,
        tokenizer_source=tokenizer_source,
        text_column=text_column,
        seq_len=seq_len,
        loss_mode=loss_mode,
    )
    cache_dir = cache_path_for(cache_root, expected)
    if is_valid_cache(cache_dir, expected) and not overwrite:
        return cache_dir
    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ids_buffer: list[int] = []
    mask_buffer: list[bool] = []
    num_sequences = 0
    supervised_tokens = 0
    total_tokens = 0
    with (tmp_dir / "input_ids.bin").open("ab") as input_file, (tmp_dir / "labels.bin").open("ab") as labels_file:
        for text in iter_texts(data_path, text_column, parquet_batch_rows):
            ids, mask = encode_text(tokenizer, text, loss_mode)
            ids_buffer.extend(ids)
            mask_buffer.extend(mask)
            while len(ids_buffer) >= seq_len:
                chunk_ids = ids_buffer[:seq_len]
                chunk_mask = mask_buffer[:seq_len]
                del ids_buffer[:seq_len]
                del mask_buffer[:seq_len]
                if loss_mode == "assistant_only" and not any(chunk_mask):
                    continue
                ids_array = np.asarray(chunk_ids, dtype=np.int32)
                labels_array = ids_array.copy()
                labels_array[~np.asarray(chunk_mask, dtype=np.bool_)] = -100
                ids_array.tofile(input_file)
                labels_array.tofile(labels_file)
                num_sequences += 1
                total_tokens += int(seq_len)
                supervised_tokens += int(np.count_nonzero(labels_array != -100))
                if num_sequences % 1000 == 0:
                    logger.info("Pretokenized %d sequences", num_sequences)
    if num_sequences <= 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Token cache build produced zero sequences")
    metadata = dict(expected)
    metadata.update(
        {
            "num_sequences": int(num_sequences),
            "total_tokens": int(total_tokens),
            "supervised_tokens": int(supervised_tokens),
            "input_ids_file": "input_ids.bin",
            "labels_file": "labels.bin",
        }
    )
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    return cache_dir


def resolve_token_cache(
    *,
    data_path: Path,
    tokenizer_source: str,
    text_column: str,
    seq_len: int,
    loss_mode: str,
    cache_root: Path,
) -> tuple[Path, dict[str, object]]:
    expected = expected_metadata(
        data_path=data_path,
        tokenizer_source=tokenizer_source,
        text_column=text_column,
        seq_len=seq_len,
        loss_mode=loss_mode,
    )
    return cache_path_for(cache_root, expected), expected


def iter_cached_batches(cache_dir: Path, *, batch_size: int, epochs: int) -> Iterator[dict[str, torch.Tensor]]:
    metadata = read_cache_metadata(cache_dir)
    if metadata is None:
        raise FileNotFoundError(f"Missing token cache metadata in {cache_dir}")
    num_sequences = int(metadata["num_sequences"])
    seq_len = int(metadata["seq_len"])
    input_ids = np.memmap(cache_dir / "input_ids.bin", dtype=np.int32, mode="r", shape=(num_sequences, seq_len))
    labels = np.memmap(cache_dir / "labels.bin", dtype=np.int32, mode="r", shape=(num_sequences, seq_len))
    for _epoch in range(epochs):
        for start in range(0, num_sequences - batch_size + 1, batch_size):
            end = start + batch_size
            input_batch = torch.from_numpy(np.asarray(input_ids[start:end], dtype=np.int64))
            label_batch = torch.from_numpy(np.asarray(labels[start:end], dtype=np.int64))
            yield {
                "input_ids": input_batch,
                "attention_mask": torch.ones_like(input_batch),
                "labels": label_batch,
            }

