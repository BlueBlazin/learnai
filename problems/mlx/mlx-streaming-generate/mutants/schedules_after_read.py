# BUG: uses mx.async_eval, but builds and schedules step t + 1 only after token t has been read and detokenized, so nothing runs during the detokenization.
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
    for index in range(n):
        text = detok(token.item())
        if index + 1 < n:
            next_token, next_logprob, cache = step(token.reshape(1), cache)
            mx.async_eval(next_token, next_logprob)
        yield text, logprob
        if index + 1 < n:
            token, logprob = next_token, next_logprob
