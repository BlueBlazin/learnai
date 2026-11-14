from collections.abc import Callable

import mlx.core as mx

BUCKETS = (16, 32, 64)


def make_sampler_step(model, state: dict, penalty: float) -> Callable[[mx.array], mx.array]:
    """Build a compiled function that picks the next token of a prompt of any length.

    model: an MLXTinyLM. `logits, _ = model(tokens)` maps a 1-D integer array of T
        tokens to logits of shape (T, vocab_size); row i depends on tokens[: i + 1] only.
    state: {"counts": float32 (vocab_size,)}, how often each token has been emitted.
        The caller owns the dict and may replace state["counts"] between calls.
    penalty: a Python float >= 0, subtracted from a token's logit once per earlier
        emission.
    Returns `step`. `token = step(prompt)` takes an int32 array of shape (T,) with
    1 <= T <= 64, returns the int32 scalar argmax of
    `logits[T - 1] - penalty * state["counts"]`, and replaces state["counts"] with the
    counts including that token. The model's Python body runs at most once per bucket
    (16, 32, 64 tokens), and nothing is evaluated.
    """
    raise NotImplementedError
