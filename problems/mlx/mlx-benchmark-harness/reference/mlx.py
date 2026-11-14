from collections.abc import Callable
from statistics import median

import mlx.core as mx


def benchmark(
    model, prompt: list[int], n_tokens: int, n_warmup: int, n_runs: int, clock: Callable[[], float]
) -> dict:
    prompt_tokens = mx.array(prompt)

    def run() -> tuple[float, float, list[mx.array]]:
        start = clock()
        logits, cache = model(prompt_tokens)
        token = mx.argmax(logits[-1])
        # Until this returns, the prefill is a graph and has taken no time at all.
        # mx.eval blocks until the token (and the cache it depends on) is computed.
        mx.eval(token)
        first_token = clock()
        tokens = [token]
        for _ in range(n_tokens):
            logits, cache = model(token.reshape(1), cache)
            token = mx.argmax(logits[-1])
            mx.eval(token)  # one evaluation per token: the loop being measured
            tokens.append(token)
        end = clock()
        return first_token - start, end - first_token, tokens

    # Warm-up runs are full, evaluated runs: a graph that is never computed compiles no
    # kernel and allocates no buffer. Their times are thrown away.
    for _ in range(n_warmup):
        run()
    runs = [run() for _ in range(n_runs)]
    prompt_time = median(times for times, _, _ in runs)
    generation_time = median(times for _, times, _ in runs)
    return {
        "prompt_tps": len(prompt) / prompt_time,
        "generation_tps": n_tokens / generation_time,
        # Read after the last clock reading: converting to ints is not part of a step.
        "tokens": [token.item() for token in runs[-1][2]],
    }
