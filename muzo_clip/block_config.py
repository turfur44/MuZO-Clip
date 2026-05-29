"""Block-row selection helpers."""

from __future__ import annotations

from typing import Literal

import torch

BlockRows = int | None | Literal["auto"]


def resolve_block_rows(param: torch.Tensor, block_rows: BlockRows) -> int | None:
    if block_rows != "auto":
        return block_rows
    if param.ndim != 2:
        raise ValueError("auto block rows only supports 2D tensors")
    rows = int(param.shape[0])
    cols = int(param.shape[1])
    if rows <= 1024:
        return None
    if cols <= 4096:
        return 1024
    return 512

