# BUG: returns the argmax as it comes, uint32: the token fed back on the second call has a new dtype, so the function is traced a second time, and the result is not int32.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1)
        return mx.where(token == eos_id, eos_id, best), state

    return mx.compile(step, inputs=model.state)
