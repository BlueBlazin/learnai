# BUG: a shared layer still runs its own indexer and throws the result away: the output is right, but there is one indexer call per layer instead of one per full layer.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    idx = None
    for layer, kind in enumerate(types):
        own = dsa.indexer(layer, h, k)
        if kind == "full":
            idx = own
        elif idx is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        h = h + dsa.attend(layer, h, idx)
    return h
