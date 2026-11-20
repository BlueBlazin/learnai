# BUG: a shared layer with no full layer before it attends to every visible token instead of raising ValueError.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    idx = None
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h, k)
        elif idx is None:
            steps = h.shape[0]
            everything = torch.arange(steps).expand(steps, steps)
            idx = torch.where(everything <= torch.arange(steps)[:, None], everything, -1)
        h = h + dsa.attend(layer, h, idx)
    return h
