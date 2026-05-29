"""Deterministic optional sparse parameter scheduling."""

from __future__ import annotations

import hashlib
from typing import Literal

from .parameter_filter import SelectedParameter

SparseUpdateMode = Literal["off", "round_robin"]


def select_active_parameters(
    selected: list[SelectedParameter],
    *,
    mode: SparseUpdateMode,
    groups: int,
    step_index: int,
) -> list[SelectedParameter]:
    if mode == "off":
        return selected
    if mode != "round_robin":
        raise ValueError(f"Unsupported sparse update mode: {mode}")
    if groups <= 1:
        return selected
    group_index = int(step_index) % int(groups)
    active = [item for index, item in enumerate(selected) if index % int(groups) == group_index]
    return active or selected


def names_hash(names: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

