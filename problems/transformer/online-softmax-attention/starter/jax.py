from collections.abc import Iterator

import jax

Block = tuple[jax.Array, jax.Array, jax.Array]


def blockwise_attention(q: jax.Array, kv_blocks: Iterator[Block]) -> jax.Array:
    """Scaled dot-product attention over keys and values that arrive in blocks.

    q: float32 tensor of shape (T, D), the queries.
    kv_blocks: a one-pass iterator of (k, v, mask) blocks: k is float32 (n, D), v is
        float32 (n, D_v), mask is bool (T, n) with True where the query may attend to
        the key. n differs from block to block. A block is gone once the next one has
        been requested.
    Returns a float32 tensor of shape (T, D_v): for each query, the softmax of its
    scores q . k / sqrt(D) over all visible keys of all blocks, applied to the values.
    """
    raise NotImplementedError
