"""Block-row selection helpers."""

from __future__ import annotations

from typing import Literal

import torch

BlockRows = int | None | Literal["auto", "auto_full"]


def resolve_block_rows(
    param: torch.Tensor,
    block_rows: BlockRows,
    *,
    full_block_max_elements: int = 8_388_608,
) -> int | None:
    if block_rows not in ("auto", "auto_full"):
        return block_rows
    if param.ndim != 2:
        raise ValueError("auto block rows only supports 2D tensors")
    rows = int(param.shape[0])
    cols = int(param.shape[1])
    if block_rows == "auto_full" and rows * cols <= int(full_block_max_elements):
        return None
    if rows <= 1024:
        return None
    if cols <= 4096:
        return 1024
    return 512
