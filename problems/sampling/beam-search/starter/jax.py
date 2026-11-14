import jax
import jax.numpy as jnp

from whetstone.kit import TinyLM, ops


def beam_search(
    lm: TinyLM,
    prompt: list[int],
    n_beams: int,
    max_new_tokens: int,
    eos_id: int | None,
    length_penalty: float,
) -> list[tuple[list[int], float]]:
    """Beam search with a KV cache: one batched model call per step.

    lm: the model. `logits, cache = lm(tokens)` runs a prefix, `lm(tokens, cache)`
        continues one. Batched tokens (B, T) give logits (B, T, vocab_size).
        `cache.select(rows)` returns a cache whose batch rows are the given rows of
        this one, in that order (rows may repeat).
    prompt: the prompt token ids, at least one.
    n_beams: the beam width, at least 1.
    max_new_tokens: the maximum length of a hypothesis, at least 1.
    eos_id: a hypothesis that generates this token is finished; None disables it.
    length_penalty: the exponent a in the final score, sum of log-probs / length**a.
    Returns the n_beams best finished hypotheses, best first, as a list of
    (tokens, score): the generated tokens as Python ints and the score as a float.
    """
    raise NotImplementedError
