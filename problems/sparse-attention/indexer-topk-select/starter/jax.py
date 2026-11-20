import jax
import jax.numpy as jnp


def index_scores(q_idx: jax.Array, k_idx: jax.Array, w: jax.Array) -> jax.Array:
    """Index scores of T queries against S keys.

    q_idx: float32, shape (T, H, D). One indexer query per query token and indexer head.
    k_idx: float32, shape (S, D). One indexer key per token, shared by all heads.
    w: float32, shape (T, H). Raw head weights of each query token; they may be negative.
    Returns float32, shape (T, S):
        I[t, s] = sum_h (w[t, h] * H**-0.5) * relu(dot(q_idx[t, h], k_idx[s]) * D**-0.5)
    """
    raise NotImplementedError


def topk_mask(scores: jax.Array, visible: jax.Array, k: int) -> jax.Array:
    """Select, in each row, the highest-scoring visible entries.

    scores: float32, shape (T, S), finite.
    visible: bool, shape (T, S). The entries row t may select.
    k: int >= 1; it may exceed S.
    Returns bool, shape (T, S): True at the min(k, number of visible entries) visible
    entries of each row with the highest scores; equal scores are ranked by index,
    lowest first.
    """
    raise NotImplementedError
