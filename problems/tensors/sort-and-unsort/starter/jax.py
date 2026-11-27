import jax
import jax.numpy as jnp


def rank_descending(x: jax.Array) -> jax.Array:
    """The indices that sort every row from largest to smallest.

    x: float32, shape (B, V), finite.
    Returns an integer tensor, shape (B, V): out[b, k] is the column of the k-th
    largest value of row b. Equal values are ranked by column, lowest first.
    """
    raise NotImplementedError


def inverse_permutation(perm: jax.Array) -> jax.Array:
    """The permutation that undoes each row of perm.

    perm: int32, shape (B, V); every row is a permutation of 0 .. V - 1.
    Returns an integer tensor, shape (B, V), with out[b, perm[b, k]] = k.
    """
    raise NotImplementedError


def keep_to_original_order(keep_sorted: jax.Array, order: jax.Array) -> jax.Array:
    """Move a decision made in sorted order back to token order.

    keep_sorted: bool, shape (B, V); keep_sorted[b, k] is the decision for rank k.
    order: int32, shape (B, V); order[b, k] is the token at rank k of row b, and every
    row is a permutation of 0 .. V - 1.
    Returns bool, shape (B, V), with out[b, order[b, k]] = keep_sorted[b, k].
    """
    raise NotImplementedError
