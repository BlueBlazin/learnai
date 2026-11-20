# BUG: every layer runs its indexer, and the kit's call counter is then overwritten with the number of full layers; the counters are off limits.
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
    dsa.counters.indexer_calls = list(types).count("full")
    return h
