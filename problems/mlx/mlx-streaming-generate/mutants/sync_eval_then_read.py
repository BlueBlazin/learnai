# BUG: evaluates each step with a blocking mx.eval and only then builds the next one, so the GPU is idle while Python detokenizes and the consumer works.
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
    for index in range(n):
        mx.eval(token, logprob)
        yield detok(token.item()), logprob
        if index + 1 < n:
            token, logprob, cache = step(token.reshape(1), cache)
