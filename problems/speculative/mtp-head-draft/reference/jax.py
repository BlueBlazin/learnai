import jax
import jax.numpy as jnp


def mtp_draft(
    hidden: jax.Array,
    heads: list[tuple[jax.Array, jax.Array, jax.Array]],
    w_out: jax.Array,
    k: int,
) -> jax.Array:
    # Position t + 1 needs no extra head: it is the model's ordinary next token.
    tokens = [jnp.argmax(w_out @ hidden)]
    for w1, b1, w2 in heads[:k]:
        # One residual block, x + SiLU(W1 x + b1), on the *base* hidden state: the heads
        # are parallel, none reads another head's output or any drafted token.
        z = w1 @ hidden + b1
        state = hidden + z * jax.nn.sigmoid(z)
        # Each head has its own unembedding.
        tokens.append(jnp.argmax(w2 @ state))
    return jnp.stack(tokens)


def mtp_accept(draft: jax.Array, target_argmax: jax.Array) -> int:
    # draft[j] was a guess for the position that target_argmax[j - 1] scores: the
    # token after draft[j - 1]. Shift by one, then keep the leading run of matches.
    match = (draft[1:] == target_argmax[:-1]).astype(jnp.int32)
    # draft[0] is the model's own token and always counts.
    return 1 + int(jnp.sum(jnp.cumprod(match)))
