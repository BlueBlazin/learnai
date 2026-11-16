import torch

from learnai.kit import TinyLM


def batched_greedy(
    lm: TinyLM, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    """Greedy generation for a batch of equal-length prompts, one batched call per step.

    lm: the model. `logits, cache = lm(tokens)` runs whole prefixes and
        `logits, cache = lm(tokens, cache)` continues from a cache. For batched
        tokens of shape (B, T), `logits` has shape (B, T, vocab_size).
    prompts: B prompts of P token ids each (a list of B lists).
    max_new_tokens: the maximum number of tokens to generate per row, at least 1.
    eos_id: a row stops after generating this token; None disables stopping.
    Returns one list of Python ints per row: that row's generated tokens (without
    the prompt), ending with eos_id if the row generated it.
    """
    raise NotImplementedError
