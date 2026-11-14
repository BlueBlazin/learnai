import jax
import jax.numpy as jnp


def count_generated(tokens: jax.Array, prompt_lengths: jax.Array, vocab_size: int) -> jax.Array:
    """Count how often each token was generated, ignoring the prompt.

    tokens: integer array of shape (B, T) with values in [0, vocab_size): each row is
        a prompt followed by the tokens generated so far.
    prompt_lengths: integer array of shape (B,); the first prompt_lengths[b] entries
        of row b are the prompt, the rest were generated.
    Returns an integer array of shape (B, vocab_size): counts[b, v] is the number of
    generated positions of row b that hold token v.
    """
    raise NotImplementedError


def apply_penalties(
    logits: jax.Array,
    token_counts: jax.Array,
    frequency_penalty: float,
    presence_penalty: float,
) -> jax.Array:
    """Subtract the OpenAI frequency and presence penalties from the logits.

    logits: float32 array of shape (B, V); entries may be -inf (masked).
    token_counts: integer array of shape (B, V), as returned by count_generated.
    Returns a float32 array of shape (B, V):
    logits - token_counts * frequency_penalty - (token_counts > 0) * presence_penalty.
    """
    raise NotImplementedError
