# BUG: there is no check for a shared layer with no full layer before it: the missing selection is passed to dsa.attend, and the ValueError comes from the kit refusing the call instead of from the function.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    idx = None
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h, k)
        h = h + dsa.attend(layer, h, idx)
    return h
