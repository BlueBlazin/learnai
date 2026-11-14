# BUG: never looks at eos_id: a row that has finished keeps generating new tokens.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        return mx.argmax(logits, axis=-1).astype(mx.int32), state

    return mx.compile(step, inputs=model.state)
