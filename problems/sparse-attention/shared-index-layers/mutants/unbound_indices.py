# BUG: the case of a shared layer with no full layer before it is not handled: the indices variable is read before it was ever assigned, which raises UnboundLocalError instead of ValueError.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h, k)
        h = h + dsa.attend(layer, h, idx)
    return h
