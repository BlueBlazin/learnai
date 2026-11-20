from collections.abc import Sequence
from typing import Any

import torch


def forward_shared(dsa: Any, h: torch.Tensor, types: Sequence[str], k: int) -> torch.Tensor:
    """A stack of sparse-attention layers in which some layers reuse a selection.

    dsa: a TinyDSA with dsa.n_layers layers. For h of shape (T, d_model):
        dsa.indexer(layer, h, k)   -> (T, min(k, T)) token indices, -1 in unused slots;
                                      counts one indexer call
        dsa.attend(layer, h, idx)  -> (T, d_model): the layer's attention over the
                                      tokens listed in idx
    h: float32, shape (T, d_model). The input of layer 0.
    types: dsa.n_layers strings, each "full" or "shared".
    k: int >= 1. How many tokens each query selects.
    Returns float32, shape (T, d_model): the output of the last layer, where every layer
    computes h = h + attend(layer, h, idx).
    Raises ValueError if a shared layer has no full layer before it.
    """
    raise NotImplementedError
