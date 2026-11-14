# BUG: pads on the left and reads the last position: the prompt is shifted to later positions and attends to the padding before it.
from collections.abc import Callable

import mlx.core as mx

BUCKETS = (16, 32, 64)


def make_sampler_step(model, state: dict, penalty: float) -> Callable[[mx.array], mx.array]:
    def pick(tokens: mx.array, last: mx.array) -> mx.array:
        logits, _ = model(tokens)
        adjusted = mx.take(logits, last, axis=0) - penalty * state["counts"]
        token = mx.argmax(adjusted).astype(mx.int32)
        state["counts"] = state["counts"] + (mx.arange(adjusted.shape[0]) == token)
        return token

    compiled = mx.compile(pick, inputs=[model.state, state], outputs=state)

    def step(prompt: mx.array) -> mx.array:
        length = prompt.shape[0]
        bucket = next(size for size in BUCKETS if size >= length)
        padded = mx.pad(prompt, (bucket - length, 0))
        return compiled(padded, mx.array(bucket - 1))

    return step
