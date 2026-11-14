# BUG: recovers the prompt's length inside the compiled function by counting nonzero tokens, but token 0 is an ordinary token that prompts contain.
from collections.abc import Callable

import mlx.core as mx

BUCKETS = (16, 32, 64)


def make_sampler_step(model, state: dict, penalty: float) -> Callable[[mx.array], mx.array]:
    def pick(tokens: mx.array) -> mx.array:
        logits, _ = model(tokens)
        last = mx.maximum((tokens != 0).sum() - 1, 0)
        adjusted = mx.take(logits, last, axis=0) - penalty * state["counts"]
        token = mx.argmax(adjusted).astype(mx.int32)
        state["counts"] = state["counts"] + (mx.arange(adjusted.shape[0]) == token)
        return token

    compiled = mx.compile(pick, inputs=[model.state, state], outputs=state)

    def step(prompt: mx.array) -> mx.array:
        length = prompt.shape[0]
        bucket = next(size for size in BUCKETS if size >= length)
        return compiled(mx.pad(prompt, (0, bucket - length)))

    return step
