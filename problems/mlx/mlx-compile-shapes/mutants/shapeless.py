# BUG: avoids retracing with shapeless=True instead of padding; the model's body builds its mask and positions from the traced length, so the first prompt of another length fails.
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

    compiled = mx.compile(pick, inputs=[model.state, state], outputs=state, shapeless=True)

    def step(prompt: mx.array) -> mx.array:
        return compiled(prompt, mx.array(prompt.shape[0] - 1))

    return step
