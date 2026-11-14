# BUG: the warm-up runs build their graphs and never evaluate them, so nothing is warmed up and the measured runs are the cold ones.
from collections.abc import Callable
from statistics import median

import mlx.core as mx


def benchmark(
    model, prompt: list[int], n_tokens: int, n_warmup: int, n_runs: int, clock: Callable[[], float]
) -> dict:
    prompt_tokens = mx.array(prompt)

    def run(evaluate: bool = True) -> tuple[float, float, list[mx.array]]:
        start = clock()
        logits, cache = model(prompt_tokens)
        token = mx.argmax(logits[-1])
        if evaluate:
            mx.eval(token)
        first_token = clock()
        tokens = [token]
        for _ in range(n_tokens):
            logits, cache = model(token.reshape(1), cache)
            token = mx.argmax(logits[-1])
            if evaluate:
                mx.eval(token)
            tokens.append(token)
        end = clock()
        return first_token - start, end - first_token, tokens

    for _ in range(n_warmup):
        run(evaluate=False)
    runs = [run() for _ in range(n_runs)]
    prompt_time = median(times for times, _, _ in runs)
    generation_time = median(times for _, times, _ in runs)
    return {
        "prompt_tps": len(prompt) / prompt_time,
        "generation_tps": n_tokens / generation_time,
        "tokens": [token.item() for token in runs[-1][2]],
    }
