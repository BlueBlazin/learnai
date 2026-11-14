# BUG: compiles without inputs=model.state, so the parameters are baked into the graph as constants at trace time and later updates of the model are ignored.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        return mx.where(token == eos_id, eos_id, best), state

    return mx.compile(step)
