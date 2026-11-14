from collections.abc import Callable

import mlx.core as mx


def make_step(model, eos_id: int) -> Callable[[mx.array, dict], tuple[mx.array, dict]]:
    """Build the compiled greedy decode step of `model`.

    model: an MLXTinyLM. `logits, state = model.step(token, state)` feeds one token per
        row: token (B,) int32, logits (B, vocab_size) float32, and `state` is the dict
        of `model.init_state(max_len, batch_size)`, whose shapes never change.
    eos_id: the end-of-sequence token; a row that is fed `eos_id` emits `eos_id`.
    Returns `step`, with `next_token, state = step(token, state)`: next_token is (B,)
    int32, the argmax of the logits (or `eos_id`), and state is the advanced state.
    `step` is traced once per input shape, reads the model's current parameters on
    every call, and evaluates nothing.
    """
    raise NotImplementedError
