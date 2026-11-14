# BUG: handles the end of sequence in a Python wrapper that reads the token with .item(): an implicit evaluation on every call, and an error for a batch of more than one row.
from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        return mx.argmax(logits, axis=-1).astype(mx.int32), state

    compiled = mx.compile(step, inputs=model.state)

    def step_with_eos(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        next_token, state = compiled(token, state)
        if token.item() == eos_id:
            next_token = token
        return next_token, state

    return step_with_eos
