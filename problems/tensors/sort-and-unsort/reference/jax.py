import jax
import jax.numpy as jnp


def rank_descending(x: jax.Array) -> jax.Array:
    # jnp.argsort is stable: equal values keep their index order. Sorting the negated
    # values ascending is a descending sort that still ranks ties lowest index first.
    return jnp.argsort(-x, axis=-1)


def inverse_permutation(perm: jax.Array) -> jax.Array:
    # perm[b, k] is where rank k points. The inverse answers the opposite question,
    # inv[b, perm[b, k]] = k, which is a write of the ranks 0 .. V - 1 through perm.
    rows = jnp.arange(perm.shape[0])[:, None]
    ranks = jnp.arange(perm.shape[1], dtype=perm.dtype)  # broadcast over the rows
    return jnp.zeros_like(perm).at[rows, perm].set(ranks)


def keep_to_original_order(keep_sorted: jax.Array, order: jax.Array) -> jax.Array:
    # order[b, k] is the token at rank k, so the decision for rank k belongs at column
    # order[b, k]: a write through the permutation. A read through it would sort again.
    rows = jnp.arange(order.shape[0])[:, None]
    return jnp.zeros_like(keep_sorted).at[rows, order].set(keep_sorted)
