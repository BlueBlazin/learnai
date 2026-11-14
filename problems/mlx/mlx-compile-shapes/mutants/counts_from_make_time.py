# BUG: passes the counts as an argument but takes them from a variable bound when the sampler was made, so an array the caller puts into state['counts'] later is never read.
from collections.abc import Callable

import mlx.core as mx

BUCKETS = (16, 32, 64)


def make_sampler_step(model, state: dict, penalty: float) -> Callable[[mx.array], mx.array]:
    def pick(tokens: mx.array, last: mx.array, counts: mx.array) -> tuple[mx.array, mx.array]:
        logits, _ = model(tokens)
        adjusted = mx.take(logits, last, axis=0) - penalty * counts
        token = mx.argmax(adjusted).astype(mx.int32)
        return token, counts + (mx.arange(adjusted.shape[0]) == token)

    compiled = mx.compile(pick, inputs=model.state)
    counts = state["counts"]

    def step(prompt: mx.array) -> mx.array:
        nonlocal counts
        length = prompt.shape[0]
        bucket = next(size for size in BUCKETS if size >= length)
        padded = mx.pad(prompt, (0, bucket - length))
        token, counts = compiled(padded, mx.array(length - 1), counts)
        state["counts"] = counts
        return token

    return step
