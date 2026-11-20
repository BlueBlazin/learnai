from typing import Any

import jax
import jax.numpy as jnp


def gdn_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_g: jax.Array,
    beta: jax.Array,
    memory: Any,
    chunk_size: int,
) -> tuple[jax.Array, Any]:
    """The gated delta rule over T tokens, sequential over chunks only.

        q, k: float32, shape (T, H, Dk). v: float32, shape (T, H, Dv).
        log_g: float32, shape (T, H), <= 0. The log of the decay of each token and head.
        beta: float32, shape (T, H), in [0, 1].
    memory: the memory before the first token, an object (see the statement):
            memory.read(rows)                   rows (H, R, Dk) -> rows @ S^T, (H, R, Dv)
            memory.update(decay, values, keys)  -> a new memory, decay * S + values^T keys
            memory.max_rows                     the most rows one call accepts
        chunk_size: int >= 1; T need not be a multiple of it.
        Returns (Y, memory): Y float32 of shape (T, H, Dv), the read of every token, and
        the memory after the last token, as returned by the last update. Both equal what
        T calls of the one-token rule with g = exp(log_g) give.
    """
    raise NotImplementedError
