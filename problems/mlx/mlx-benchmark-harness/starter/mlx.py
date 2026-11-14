from collections.abc import Callable
from statistics import median

import mlx.core as mx


def benchmark(
    model, prompt: list[int], n_tokens: int, n_warmup: int, n_runs: int, clock: Callable[[], float]
) -> dict:
    """Prompt and generation throughput of `model`, measured over repeated runs.

    model: an MLXTinyLM-like model. `logits, cache = model(tokens)` and
        `logits, cache = model(tokens, cache)` return lazy arrays.
    prompt: list of P >= 1 ints.
    n_tokens: decode steps per run after the prompt, n_tokens >= 1.
    n_warmup: runs to perform and discard first, n_warmup >= 0.
    n_runs: runs to measure, n_runs >= 1.
    clock: clock() -> float, the current time in seconds. Read it three times per run:
        before the prompt, after the first token is computed, after the last one is.
    Returns {"prompt_tps": float, "generation_tps": float, "tokens": list of
    n_tokens + 1 ints}: P and n_tokens divided by the median prompt time and the
    median generation time of the measured runs, and the tokens of a run.
    """
    raise NotImplementedError
