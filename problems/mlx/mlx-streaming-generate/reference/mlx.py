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
    # Returns at once: the GPU starts on the prefill while Python carries on.
    mx.async_eval(token, logprob)
    for index in range(n):
        if index + 1 < n:
            # Build step index + 1 from the token *array* (its value is not needed to
            # build a graph) and schedule it before reading token `index`.
            next_token, next_logprob, cache = step(token.reshape(1), cache)
            mx.async_eval(next_token, next_logprob)
        # The one read per token. It waits for step `index` only; step index + 1 is
        # already queued and runs while Python detokenizes and the consumer works.
        yield detok(token.item()), logprob
        if index + 1 < n:
            token, logprob = next_token, next_logprob
