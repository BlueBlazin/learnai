import jax
import jax.numpy as jnp


def index_scores(q_idx: jax.Array, k_idx: jax.Array, w: jax.Array) -> jax.Array:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    # One dot product per (query, head, key); the ReLU acts on each head separately.
    per_head = jnp.maximum(jnp.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5, 0.0)
    # The head weights keep their sign: a head can lower a score.
    return jnp.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: jax.Array, visible: jax.Array, k: int) -> jax.Array:
    # Invisible entries go last, whatever their score.
    masked = jnp.where(visible, scores, -jnp.inf)
    # jnp.argsort is stable: ascending in -masked ranks equal scores by index, lowest first.
    order = jnp.argsort(-masked, axis=-1)
    rank = jnp.argsort(order, axis=-1)  # rank[t, s]: the place of entry s in row t
    # Rows with fewer than k visible entries rank some invisible ones below k.
    return (rank < k) & visible
