# BUG: the selections of all full layers are computed up front from the input of the stack instead of from each full layer's own input.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    if types and types[0] != "full":
        raise ValueError("layer 0 is shared, but no full layer comes before it")
    chosen = {i: dsa.indexer(i, h, k) for i, kind in enumerate(types) if kind == "full"}
    idx = None
    for layer in range(len(types)):
        idx = chosen.get(layer, idx)
        h = h + dsa.attend(layer, h, idx)
    return h
