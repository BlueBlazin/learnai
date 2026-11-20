from collections.abc import Sequence
from typing import Any

import jax


def forward_shared(dsa: Any, h: jax.Array, types: Sequence[str], k: int) -> jax.Array:
    idx = None  # the selection of the nearest full layer so far
    for layer, kind in enumerate(types):
        if kind == "full":
            # This layer selects for itself, from its own input.
            idx = dsa.indexer(layer, h, k)
        elif idx is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        # A shared layer attends with the selection it inherited; it has no indexer.
        h = h + dsa.attend(layer, h, idx)
    return h
