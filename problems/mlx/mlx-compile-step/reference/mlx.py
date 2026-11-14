from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    def step(token: mx.array, state: dict) -> tuple[mx.array, dict]:
        logits, state = model.step(token, state)
        # argmax returns uint32. Cast back to the dtype of the input, or the token fed
        # back on the second call has a new dtype and the function is traced again.
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        # A finished row stays finished. This is a branch on a value, so it has to be
        # in the graph: Python's `if` runs once, at trace time, on a placeholder.
        next_token = mx.where(token == eos_id, eos_id, best)
        return next_token, state

    # The parameters are read by `model.step` without being arguments. Declared as
    # inputs they are looked up at every call; undeclared, the values they had at trace
    # time would be baked into the compiled graph as constants.
    return mx.compile(step, inputs=model.state)
