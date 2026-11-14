import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    """Greedy generation of `n` tokens with one `mx.eval` per generated token.

    model: an MLXTinyLM. `logits, cache = model(tokens)` runs a 1-D integer array of
        tokens from scratch, `logits, cache = model(tokens, cache)` continues a cache;
        `logits` has shape (T, vocab_size) and both results are lazy.
    prompt: list of P >= 1 ints.
    n: number of tokens to generate, n >= 1.
    Returns (tokens, logprob): a list of n scalar integer arrays (the generated
    tokens) and a scalar float32 array (the sum of their log-probabilities), all of
    them already evaluated.
    """
    raise NotImplementedError
