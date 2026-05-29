"""Parameter selection rules for MuZO-Clip hidden 2D matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

DEFAULT_TRAINABLE_SUBSTRINGS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_FROZEN_SUBSTRINGS = (
    "embed_tokens",
    "lm_head",
    "norm",
    "bias",
    "layernorm",
    "ln_",
)


@dataclass(frozen=True)
class SelectedParameter:
    name: str
    param: torch.nn.Parameter


def should_select_parameter(
    name: str,
    param: torch.Tensor,
    trainable_substrings: Iterable[str] = DEFAULT_TRAINABLE_SUBSTRINGS,
    frozen_substrings: Iterable[str] = DEFAULT_FROZEN_SUBSTRINGS,
) -> bool:
    """Return True for default Muon-style hidden 2D weight matrices."""

    lowered = name.lower()
    return (
        param.ndim == 2
        and any(token.lower() in lowered for token in trainable_substrings)
        and not any(token.lower() in lowered for token in frozen_substrings)
    )


def select_muzo_parameters(
    model: torch.nn.Module,
    trainable_substrings: Iterable[str] = DEFAULT_TRAINABLE_SUBSTRINGS,
    frozen_substrings: Iterable[str] = DEFAULT_FROZEN_SUBSTRINGS,
) -> list[SelectedParameter]:
    """Collect parameters that MuZO-Clip may update in-place."""

    selected: list[SelectedParameter] = []
    for name, param in model.named_parameters():
        if should_select_parameter(name, param, trainable_substrings, frozen_substrings):
            selected.append(SelectedParameter(name=name, param=param))
    return selected
