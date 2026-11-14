# BUG: marks a row as finished when the new token is eos_id and not when the token fed in is: the same on the step that emits it, but the next step generates again.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        return mx.where(best == eos_id, eos_id, best), state

    return mx.compile(step, inputs=model.state)
