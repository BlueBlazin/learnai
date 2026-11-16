import mlx.core as mx

from learnai.kit import KVCache, TinyLM


def generate_with_store(
    lm: TinyLM, store: dict[tuple[int, ...], KVCache], prompt: list[int], max_new_tokens: int
) -> tuple[list[int], dict[tuple[int, ...], KVCache]]:
    """Greedy generation that starts from the best cached prefix in `store`.

    lm: the model. `logits, cache = lm(tokens)` starts a sequence;
        `logits, cache = lm(tokens, cache)` continues the cached prefix.
    store: {tuple of token ids: KVCache covering exactly those tokens}; may be empty.
        `cache.truncate(n)` returns a cache covering only the first n tokens.
    prompt: the prompt token ids, at least one.
    max_new_tokens: the number of tokens to generate, at least 1.
    Returns (tokens, store): the generated tokens as a list of max_new_tokens Python
    ints, and a new store with every old entry plus tuple(prompt) mapped to a cache
    covering exactly the prompt.
    """
    raise NotImplementedError
