# BUG: a shared layer runs the nearest full layer's indexer again on its own input instead of reusing the indices that layer selected: other indices, and an indexer call per layer.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    nearest = None
    for layer, kind in enumerate(types):
        if kind == "full":
            nearest = layer
        elif nearest is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        idx = dsa.indexer(nearest, h, k)
        h = h + dsa.attend(layer, h, idx)
    return h
