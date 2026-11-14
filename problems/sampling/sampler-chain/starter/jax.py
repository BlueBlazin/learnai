import jax
import jax.numpy as jnp

from whetstone.kit import ops


def chain_logits(logits: jax.Array, cfg: dict, history: jax.Array) -> jax.Array:
    """The logits after penalties, top-k, top-p, min-p, and temperature, in that order.

    logits: (B, V) float32 next-token logits; entries may be -inf (masked).
    cfg: a dict with the keys "repetition_penalty", "frequency_penalty",
        "presence_penalty", "top_k", "top_p", "min_p", "temperature".
    history: (B, T) integer tensor, the tokens each row has produced so far.
    Returns (B, V) float32: removed tokens are -inf, the others are the penalized
    logits divided by the temperature.
    """
    raise NotImplementedError


def sample_chain(logits: jax.Array, cfg: dict, history: jax.Array, u: jax.Array) -> jax.Array:
    """One token per row, by inverse-CDF sampling from softmax(chain_logits(...)).

    u: (B,) float32 uniform numbers in [0, 1), one per row.
    Returns a (B,) integer tensor of token ids.
    """
    raise NotImplementedError
