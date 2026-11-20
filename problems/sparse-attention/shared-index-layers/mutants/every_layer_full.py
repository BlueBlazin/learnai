# BUG: the layer types are ignored: every layer runs its own indexer and attends with its own selection.
from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    for layer, _kind in enumerate(types):
        idx = dsa.indexer(layer, h, k)
        h = h + dsa.attend(layer, h, idx)
    return h
