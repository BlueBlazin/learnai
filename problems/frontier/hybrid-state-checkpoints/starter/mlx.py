import mlx.core as mx

from learnai.kit import RecurrentLM, RecurrentState

Store = dict[tuple[int, ...], RecurrentState]


def generate_with_checkpoints(
    lm: RecurrentLM,
    store: Store,
    prompt: list[int],
    max_new_tokens: int,
    interval: int,
    max_checkpoints: int,
) -> tuple[list[int], Store]:
    """Greedy generation with a recurrent-state model and a store of state checkpoints.

    lm: the model. `logits, state = lm(tokens)` starts a sequence;
        `logits, state = lm(tokens, state)` continues from a state. logits has one row
        per token of the call. A state has a fixed size and cannot be truncated.
    store: {tuple of token ids: the state after exactly those tokens}, in order of use,
        least recently used first; may be empty.
    prompt: the prompt token ids, at least one.
    max_new_tokens: the number of tokens to generate, at least 1.
    interval: checkpoints are saved at sequence lengths that are multiples of it, and at
        the end of the prompt.
    max_checkpoints: the most entries the returned store may hold, at least 1.
    Returns (tokens, store): the generated tokens as a list of max_new_tokens Python
    ints, and a new store (a new dict), least recently used first.
    """
    raise NotImplementedError
