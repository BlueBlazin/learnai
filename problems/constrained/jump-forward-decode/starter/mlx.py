import mlx.core as mx

from whetstone.kit import TinyLM, TokenConstraint


def jump_forward_decode(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    """Constrained greedy generation that runs the model only at free steps.

    lm: the model. `logits, cache = lm(tokens)` starts a sequence;
        `logits, cache = lm(tokens, cache)` continues the cached prefix with one or
        more tokens. `logits` has one row per token passed in that call, shape
        (T, vocab_size).
    prompt: the prompt token ids, at least one. The constraint does not apply to it.
    constraint: `start() -> state`, `allowed(state) -> sorted list of token ids`,
        `advance(state, token) -> state`, `is_accepting(state) -> bool`.
        `allowed` never contains `eos_id`; it is permitted exactly in accepting states.
    max_new_tokens: the maximum number of tokens to generate, at least 1; forced
        tokens count.
    eos_id: the end-of-sequence token; it ends generation.
    Returns the generated tokens (without the prompt) as a list of Python ints: the
    same tokens as constrained greedy decoding, with one model call per step at which
    two or more tokens are permitted and none for the others.
    """
    raise NotImplementedError
