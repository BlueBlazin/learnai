# BUG: the indexer is called without k and selects its default number of tokens.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    idx = None
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h)
        elif idx is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        h = h + dsa.attend(layer, h, idx)
    return h
