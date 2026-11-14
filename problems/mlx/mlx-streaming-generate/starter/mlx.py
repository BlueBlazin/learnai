from collections.abc import Callable, Iterator

import mlx.core as mx


def stream_generate(
    model, prompt: list[int], n: int, detok: Callable[[int], str]
) -> Iterator[tuple[str, mx.array]]:
    """Greedy generation as a generator that yields each token's text as soon as it exists.

    model: an MLXTinyLM. `logits, cache = model(tokens)` runs a 1-D integer array of
        tokens from scratch, `logits, cache = model(tokens, cache)` continues a cache;
        `logits` has shape (T, vocab_size) and both results are lazy.
    prompt: list of P >= 1 ints.
    n: number of tokens to generate, n >= 1.
    detok: detok(token) -> str for one Python int; called once per token, in order.
    Yields n pairs (text, logprob): the text of the token and its log-probability as a
    scalar float32 array that is already scheduled or computed. Step t + 1 is scheduled
    with mx.async_eval before token t is read with .item(); mx.eval is not used.
    """
    raise NotImplementedError
    yield  # makes this function a generator, as the solution must be
