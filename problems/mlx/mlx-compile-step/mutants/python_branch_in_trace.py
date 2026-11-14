# BUG: tests for the end of sequence with a Python `if` inside the compiled function; at trace time the token is a placeholder with no value, so the first call raises.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        if mx.all(token == eos_id).item():
            return token, state
        return mx.argmax(logits, axis=-1).astype(mx.int32), state

    return mx.compile(step, inputs=model.state)
