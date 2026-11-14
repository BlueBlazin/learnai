# BUG: wraps the compiled function and evaluates its outputs before returning, so the caller can no longer schedule the step together with other work or asynchronously.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        return mx.where(token == eos_id, eos_id, best), state

    compiled = mx.compile(step, inputs=model.state)

    def eager_step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        next_token, state = compiled(token, state)
        mx.eval(next_token, state)
        return next_token, state

    return eager_step
