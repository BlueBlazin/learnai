# BUG: calls mx.compile inside the step, so every call builds a new compiled function with an empty trace cache and traces again.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        return mx.where(token == eos_id, eos_id, best), state

    def compiled_step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        return mx.compile(step, inputs=model.state)(token, state)

    return compiled_step
