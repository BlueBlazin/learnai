# BUG: every shared layer attends with the selection of the first full layer of the stack instead of the nearest full layer before it.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    first, idx = None, None
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h, k)
            first = idx if first is None else first
        elif first is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        else:
            idx = first
        h = h + dsa.attend(layer, h, idx)
    return h
