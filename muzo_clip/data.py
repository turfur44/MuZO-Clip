"""Dataset helpers for fixed-length causal LM batches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

ASSISTANT_TAG = "<|im_start|>assistant"
IM_END_TAG = "<|im_end|>"


def iter_texts(path: Path, text_column: str, batch_rows: int) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=[text_column]):
            column = batch.column(0)
            for value in column.to_pylist():
                if value is not None:
                    text = str(value)
                    if text:
                        yield text
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text:
                yield text


def assistant_char_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    search_from = 0
    while True:
        tag_start = text.find(ASSISTANT_TAG, search_from)
        if tag_start < 0:
            break
        content_start = tag_start + len(ASSISTANT_TAG)
        if content_start < len(text) and text[content_start] == "\n":
            content_start += 1
        end_start = text.find(IM_END_TAG, content_start)
        if end_start < 0:
            spans.append((content_start, len(text)))
            break
        spans.append((content_start, end_start + len(IM_END_TAG)))
        search_from = end_start + len(IM_END_TAG)
    return spans


def token_mask_from_offsets(text: str, offsets: list[tuple[int, int]], loss_mode: str) -> list[bool]:
    if loss_mode == "full_text":
        return [end > start for start, end in offsets]
    spans = assistant_char_spans(text)
    if not spans:
        return [False for _ in offsets]
    mask: list[bool] = []
    for start, end in offsets:
        if end <= start:
            mask.append(False)
            continue
        mask.append(any(start < span_end and end > span_start for span_start, span_end in spans))
    return mask


def encode_text(tokenizer, text: str, loss_mode: str) -> tuple[list[int], list[bool]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return ids, token_mask_from_offsets(text, offsets, loss_mode)


def iter_sequences(
    tokenizer,
    data_path: Path,
    text_column: str,
    seq_len: int,
    loss_mode: str,
    parquet_batch_rows: int,
) -> Iterator[tuple[list[int], list[bool]]]:
    ids_buffer: list[int] = []
    mask_buffer: list[bool] = []
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
            yield chunk_ids, chunk_mask


def iter_batches(
    tokenizer,
    data_path: Path,
    text_column: str,
    seq_len: int,
    batch_size: int,
    epochs: int,
    loss_mode: str,
    parquet_batch_rows: int,
) -> Iterator[dict[str, torch.Tensor]]:
    for _epoch in range(epochs):
        ids_batch: list[list[int]] = []
        mask_batch: list[list[bool]] = []
        for ids, mask in iter_sequences(tokenizer, data_path, text_column, seq_len, loss_mode, parquet_batch_rows):
            ids_batch.append(ids)
            mask_batch.append(mask)
            if len(ids_batch) < batch_size:
                continue
            input_ids = torch.tensor(ids_batch, dtype=torch.long)
            labels = input_ids.clone()
            supervised = torch.tensor(mask_batch, dtype=torch.bool)
            labels.masked_fill_(~supervised, -100)
            yield {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
            ids_batch.clear()
            mask_batch.clear()

