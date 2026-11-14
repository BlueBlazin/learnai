from collections.abc import Callable

import mlx.core as mx

BUCKETS = (16, 32, 64)


def make_sampler_step(model, state: dict, penalty: float) -> Callable[[mx.array], mx.array]:
    def pick(tokens: mx.array, last: mx.array) -> mx.array:
        logits, _ = model(tokens)  # (bucket, vocab_size): the shape is fixed per bucket
        # `last` is an array, so its value is fed at every call. A Python int here
        # would be part of the signature: one trace per prompt length.
        adjusted = mx.take(logits, last, axis=0) - penalty * state["counts"]
        token = mx.argmax(adjusted).astype(mx.int32)
        # A side effect on captured state. `outputs=state` makes the compiled function
        # write the new array into the dict on every call, not only during the trace.
        state["counts"] = state["counts"] + (mx.arange(adjusted.shape[0]) == token)
        return token

    # `inputs`: arrays the function reads without receiving them (the counts, and the
    # weights), looked up at call time instead of being frozen into the graph.
    compiled = mx.compile(pick, inputs=[model.state, state], outputs=state)

    def step(prompt: mx.array) -> mx.array:
        length = prompt.shape[0]
        bucket = next(size for size in BUCKETS if size >= length)
        # Pad on the right. The model is causal, so the logits at position
        # length - 1 do not depend on what follows it.
        padded = mx.pad(prompt, (0, bucket - length))
        return compiled(padded, mx.array(length - 1))

    return step
