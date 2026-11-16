import torch

from learnai.kit import KVCache, TinyLM


def prefill(lm: TinyLM, prompt: list[int], chunk_size: int) -> tuple[torch.Tensor, KVCache]:
    """Run `prompt` through the model in chunks of at most `chunk_size` tokens.

    lm: the model. `logits, cache = lm(tokens)` starts a sequence;
        `logits, cache = lm(tokens, cache)` continues the cached prefix. `logits` has
        one row per token passed in that call, shape (T, vocab_size).
    prompt: the prompt token ids, at least one.
    chunk_size: the most tokens one call may pass, at least 1.
    Returns (last_logits, cache): the logits row of the last prompt token, a float32
    tensor of shape (vocab_size,), and the cache covering the whole prompt.
    """
    raise NotImplementedError
