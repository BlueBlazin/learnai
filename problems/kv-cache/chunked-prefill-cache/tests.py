"""Tests for chunked-prefill-cache.

Every case runs the solution on a model that records its calls, then checks, in this
order: the returned logits (against the whole prompt in one call), the returned cache
(its length, and that a further token continues from it correctly), the two counters
(`ceil(P / chunk_size)` calls and `P` tokens, exactly), and the layout of the chunks.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import KVCache, TinyLM

FN = "prefill"

# The worked example of the statement.
PROMPT = [0, 17, 42, 5, 9, 33, 12, 60, 21, 7]


class RecordingLM(TinyLM):
    """A TinyLM that also notes how many tokens each call passed."""

    def __init__(self, fw, seed):
        super().__init__(fw, seed=seed)
        self.call_sizes = []

    def __call__(self, tokens, cache=None):
        ids = self.fw.numpy(tokens) if self.fw.is_tensor(tokens) else tokens
        self.call_sizes.append(int(np.asarray(ids).size))
        return super().__call__(tokens, cache)


def run(ctx, seed, prompt, chunk_size):
    """Run the solution and check everything. Returns the logits as numpy."""
    prompt = [int(t) for t in prompt]
    oracle = TinyLM(ctx.oracle_fw, seed=seed)
    full = ctx.oracle_fw.numpy(oracle(prompt)[0])

    lm = RecordingLM(ctx.fw, seed)
    result = ctx.call(FN, lm, list(prompt), chunk_size)
    check.that(
        isinstance(result, tuple) and len(result) == 2,
        f"{FN}: must return a tuple (last_logits, cache)",
        f"got {type(result).__name__}",
    )
    last_logits, cache = result
    setup = f"prompt = {prompt}\nchunk_size = {chunk_size}\ncalls made (tokens per call): "
    setup += f"{lm.call_sizes}\n"

    # 1. The logits. Splitting the calls changes them by about 1e-5 at most.
    check.that(
        isinstance(last_logits, np.ndarray),
        "last_logits: must be a tensor, the last row of the final call's logits",
        f"got {type(last_logits).__name__}",
    )
    check.shape(last_logits, (lm.vocab_size,), label="last_logits")
    check.allclose(last_logits, full[-1], rtol=1e-3, atol=1e-4, label="last_logits", dtype=True)

    # 2. The cache: it covers the whole prompt, and decoding can go on from it.
    check.that(isinstance(cache, KVCache), "cache: must be the KVCache returned by the model")
    check.that(
        cache.length == len(prompt),
        "cache: it does not cover the whole prompt",
        f"cache.length = {cache.length}, the prompt has {len(prompt)} tokens\n{setup}",
    )
    probe = int(np.argmax(full[-1]))
    continued = ctx.oracle_fw.numpy(oracle([probe], cache)[0])[-1]
    direct = ctx.oracle_fw.numpy(oracle([*prompt, probe])[0])[-1]
    check.allclose(
        continued, direct, rtol=1e-3, atol=1e-4, label="logits of a token fed after the cache"
    )

    # 3. The cost, exactly: no token twice, no call more than needed.
    n_calls = -(-len(prompt) // chunk_size)
    check.cost(lm.counters, "tokens_processed", exactly=len(prompt), unit="tokens")
    check.cost(lm.counters, "forward_calls", exactly=n_calls)

    # 4. The layout: full chunks, then the remainder.
    sizes = [min(chunk_size, len(prompt) - start) for start in range(0, len(prompt), chunk_size)]
    check.that(
        max(lm.call_sizes) <= chunk_size,
        "a call passed more than chunk_size tokens",
        setup,
    )
    check.that(
        lm.call_sizes == sizes,
        "the chunks are not consecutive full chunks followed by the remainder",
        f"{setup}expected: {sizes}\n",
    )
    return last_logits


@case("the worked example: 10 tokens in chunks of 4 is 3 calls of 4, 4, and 2", public=True)
def worked_example(ctx):
    logits = run(ctx, 3, PROMPT, 4)
    if int(np.argmax(logits)) != 18:
        raise RuntimeError("authoring error: the statement says the argmax is 18")


@case("a prompt of 12 tokens in chunks of 4: exactly 3 calls, no empty one", public=True)
def exact_multiple(ctx):
    # Catches the trailing empty call, which only appears when chunk_size divides P.
    run(ctx, 3, [*PROMPT, 44, 8], 4)
    run(ctx, 5, [0, 9, 9, 31, 2, 50], 3)


@case("chunk_size at least the prompt length: a single call", public=True)
def one_chunk(ctx):
    run(ctx, 3, PROMPT, 10)
    run(ctx, 3, PROMPT, 64)


@case("chunk_size = 1: one call per token")
def chunk_of_one(ctx):
    run(ctx, 5, [0, 61, 13, 13, 27, 40, 3], 1)


@case("a 100-token prompt in chunks of 16 on another model: 7 calls, 100 tokens")
def long_prompt(ctx):
    # Re-feeding the prefix with every chunk costs 436 tokens here.
    run(ctx, 7, ctx.randint(0, 64, 100), 16)


@case("a single-token prompt")
def single_token(ctx):
    run(ctx, 5, [0], 4)
    run(ctx, 5, [37], 1)


@case("one token left over, and one token short of a full chunk")
def boundaries(ctx):
    # 9 = 4 + 4 + 1 and 11 = 4 + 4 + 3; 7 = 6 + 1 catches a remainder merged into the
    # last chunk and a split that puts the short chunk first.
    run(ctx, 9, ctx.randint(0, 64, 9), 4)
    run(ctx, 9, ctx.randint(0, 64, 11), 4)
    run(ctx, 11, ctx.randint(0, 64, 7), 6)


@case("random prompt lengths and chunk sizes on two models")
def random_sweep(ctx):
    for seed in (2, 4):
        for _ in range(5):
            length = int(ctx.rng.integers(1, 60))
            chunk_size = int(ctx.rng.integers(1, 24))
            run(ctx, seed, ctx.randint(0, 64, length), chunk_size)
