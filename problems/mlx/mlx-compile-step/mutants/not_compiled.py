# BUG: returns the plain Python function: correct tokens, but the model's Python body runs on every call and nothing is compiled.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        return mx.where(token == eos_id, eos_id, best), state

    return step
