# BUG: takes the middle element of the sorted times, which for an even number of runs is the upper of the two middle values and not their mean.
from collections.abc import Callable

import mlx.core as mx


def benchmark(
    model, prompt: list[int], n_tokens: int, n_warmup: int, n_runs: int, clock: Callable[[], float]
) -> dict:
    prompt_tokens = mx.array(prompt)

    def run() -> tuple[float, float, list[mx.array]]:
        start = clock()
        logits, cache = model(prompt_tokens)
        token = mx.argmax(logits[-1])
        mx.eval(token)
        first_token = clock()
        tokens = [token]
        for _ in range(n_tokens):
            logits, cache = model(token.reshape(1), cache)
            token = mx.argmax(logits[-1])
            mx.eval(token)
            tokens.append(token)
        end = clock()
        return first_token - start, end - first_token, tokens

    for _ in range(n_warmup):
        run()
    runs = [run() for _ in range(n_runs)]
    prompt_time = sorted(times for times, _, _ in runs)[n_runs // 2]
    generation_time = sorted(times for _, times, _ in runs)[n_runs // 2]
    return {
        "prompt_tps": len(prompt) / prompt_time,
        "generation_tps": n_tokens / generation_time,
        "tokens": [token.item() for token in runs[-1][2]],
    }
