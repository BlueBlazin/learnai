import mlx.core as mx

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    """Greedy generation with the KV cache: every token goes through the model once.

    lm: the model. `logits, cache = lm(tokens)` runs a whole prefix;
        `logits, cache = lm(tokens, cache)` continues from a cache. `logits` has one
        row per token passed in that call, shape (T, vocab_size).
    prompt: the prompt token ids, at least one.
    max_new_tokens: the maximum number of tokens to generate, at least 1.
    eos_id: generation stops after this token is generated; None disables stopping.
    Returns the generated tokens (without the prompt) as a list of Python ints,
    ending with eos_id if it was generated.
    """
    raise NotImplementedError
