import torch

from learnai.kit import TinyLM, TokenConstraint


def constrained_greedy(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    """Greedy generation in which every token is permitted by `constraint`.

    lm: the model. `logits, cache = lm(tokens)` starts a sequence;
        `logits, cache = lm(tokens, cache)` continues the cached prefix. `logits` has
        one row per token passed in that call, shape (T, vocab_size).
    prompt: the prompt token ids, at least one. The constraint does not apply to it.
    constraint: `start() -> state`, `allowed(state) -> sorted list of token ids`,
        `advance(state, token) -> state`, `is_accepting(state) -> bool`.
        `allowed` never contains `eos_id`.
    max_new_tokens: the maximum number of tokens to generate, at least 1.
    eos_id: the end-of-sequence token. It may be generated only in an accepting
        state, and ends generation.
    Returns the generated tokens (without the prompt) as a list of Python ints,
    ending with eos_id if it was generated. The list is empty if no token is
    permitted at the start.
    """
    raise NotImplementedError
