# BUG: always schedules the next step, including after the final token, so one forward pass per generation is computed and thrown away.
from collections.abc import Callable, Iterator

import mlx.core as mx


def stream_generate(
    model, prompt: list[int], n: int, detok: Callable[[int], str]
) -> Iterator[tuple[str, mx.array]]:
    def step(tokens: mx.array, cache: list | None) -> tuple[mx.array, mx.array, list]:
        logits, cache = model(tokens, cache)
        logprobs = logits[-1] - mx.logsumexp(logits[-1])
        token = mx.argmax(logprobs)
        return token, logprobs[token], cache

    token, logprob, cache = step(mx.array(prompt), None)
    mx.async_eval(token, logprob)
    for _ in range(n):
        next_token, next_logprob, cache = step(token.reshape(1), cache)
        mx.async_eval(next_token, next_logprob)
        yield detok(token.item()), logprob
        token, logprob = next_token, next_logprob
