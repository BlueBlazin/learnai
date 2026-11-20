# BUG: each layer's attention output replaces the hidden state instead of being added to it.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    idx = None
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h, k)
        elif idx is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        h = dsa.attend(layer, h, idx)
    return h
