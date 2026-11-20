from typing import Any

import jax
import jax.numpy as jnp


def sparse_attend(q: jax.Array, store: Any, idx: jax.Array) -> jax.Array:
    valid = idx >= 0  # (T, k): which slots hold a token
    # Read the selected rows only; an unused slot is not read and comes back as zeros.
    k_rows, v_rows = store.gather(idx, valid=valid)  # (T, k, H, d) each
    scores = jnp.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    # A zero row would score 0 and take weight: unused slots must score -inf.
    scores = jnp.where(valid[:, None, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    # A row with no token at all is NaN after the softmax; it attends to nothing.
    weights = jnp.where(valid[:, None, :], weights, 0.0)
    return jnp.einsum("thj,tjhd->thd", weights, v_rows)
