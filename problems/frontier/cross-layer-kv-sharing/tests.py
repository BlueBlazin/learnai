"""Tests for cross-layer-kv-sharing.

`run` drives one sequence: `prefill_shared` on the prompt, then `decode_step` for each
further token. After every call it checks, in this order, the next-token logits
(against the kit's forward pass of the sharing model on the whole sequence, without a
cache), the shape of the returned cache (layers below `share_from` hold every token,
the layers from `share_from` up hold none), and the cost of that call:

    prefill_shared   layer_tokens <= s * P + (L - s)    layer_calls <= L
    decode_step      layer_tokens <= L                  layer_calls <= L

The tokens that are fed come from the test (the greedy tokens of the sharing model),
so a solution's wrong logits cannot change what it is asked next.
"""

from learnai.judge import case, check
from learnai.kit import LayerCache, LayeredLM

PROMPT = [0, 17, 42, 5]
EXAMPLE_FIRST = 48  # the sharing model's greedy token after PROMPT (seed 3, share_from 2)
TOL = {"rtol": 1e-3, "atol": 1e-4}  # cached and uncached passes agree to about 1e-5


def kv_bytes(cache):
    """Bytes of every key and value array the cache holds."""
    arrays = [a for a in (*cache.keys, *cache.values) if a is not None]
    return sum(a.nbytes for a in arrays)


def check_call(ctx, lm, name, result, want, n_tokens, share_from, limit_tokens):
    check.that(
        isinstance(result, tuple) and len(result) == 2,
        f"{name}: must return a tuple (logits, cache)",
        f"got {type(result).__name__}",
    )
    logits, cache = result
    check.shape(logits, (lm.vocab_size,), label=f"{name}: logits")
    check.allclose(logits, want, label=f"{name}: next-token logits", **TOL)
    check.that(
        isinstance(cache, LayerCache),
        f"{name}: the second result must be the cache that lm.layers returned",
        f"got {type(cache).__name__}",
    )
    lengths = cache.lengths
    upper = [i for i in range(share_from, lm.n_layers) if lengths[i]]
    check.that(
        not upper,
        f"{name}: layer {upper[0] if upper else 0} stores keys and values of its own, "
        "but the layers from share_from up read those of layer share_from - 1",
        f"cache.lengths = {lengths}, expected {(n_tokens,) * share_from} then zeros",
    )
    check.that(
        all(n == n_tokens for n in lengths[:share_from]),
        f"{name}: the layers below share_from have cached {lengths[:share_from]} tokens, "
        "which is not the number of tokens processed so far",
        f"cache.lengths = {lengths}, expected {(n_tokens,) * share_from} then zeros",
    )
    kind = "prefill" if name == "prefill_shared" else "decode"
    check.cost(
        lm.counters,
        "layer_tokens",
        at_most=limit_tokens,
        unit="layer-tokens",
        metric=f"{kind}_layer_tokens",
    )
    check.cost(
        lm.counters,
        "layer_calls",
        at_most=lm.n_layers,
        unit="layer calls",
        metric=f"{kind}_layer_calls",
    )
    return cache


def run(ctx, seed, prompt, share_from, steps, n_layers=4, expected_first=None):
    """Prefill, then `steps` decode steps. Returns the final cache."""
    oracle = LayeredLM(ctx.oracle_fw, seed=seed, n_layers=n_layers)
    lm = LayeredLM(ctx.fw, seed=seed, n_layers=n_layers)

    def definition(seq):
        return ctx.oracle_fw.numpy(oracle(seq, share_from=share_from)[0])[-1]

    seq = list(prompt)
    want = definition(seq)
    if expected_first is not None and int(want.argmax()) != expected_first:
        raise RuntimeError(f"authoring error: the first token is {int(want.argmax())}")
    result = ctx.call("prefill_shared", lm, list(prompt), share_from)
    limit = share_from * len(prompt) + (n_layers - share_from)
    cache = check_call(ctx, lm, "prefill_shared", result, want, len(seq), share_from, limit)
    for _ in range(steps):
        token = int(want.argmax())
        seq.append(token)
        want = definition(seq)
        lm.counters.reset()
        before = cache.lengths
        result = ctx.call("decode_step", lm, token, cache, share_from)
        check.that(cache.lengths == before, "decode_step modified the cache it was given")
        cache = check_call(ctx, lm, "decode_step", result, want, len(seq), share_from, n_layers)
    return lm, cache


@case("the worked example: 4 prompt tokens, 4 layers, share_from = 2", public=True)
def worked_example(ctx):
    # Prefill costs 2 * 4 + 2 = 10 layer-tokens instead of 16.
    run(ctx, 3, PROMPT, 2, 3, expected_first=EXAMPLE_FIRST)


@case("share_from = 1 and share_from = 3: one stored layer, and one reading layer", public=True)
def other_split_points(ctx):
    # With share_from = 1 the source layer is layer 0; with 3 it is layer 2.
    run(ctx, 3, PROMPT, 1, 3)
    run(ctx, 3, PROMPT, 3, 3)


@case("a 24-token prompt: the upper layers run on the last position only", public=True)
def long_prompt(ctx):
    # 2 * 24 + 2 = 50 layer-tokens. The full stack on every token would take 96.
    prompt = [0, *ctx.randint(2, 64, 23).tolist()]
    run(ctx, 5, prompt, 2, 2)


@case("stored KV after 20 tokens: share_from layers, not all of them", public=True)
def stored_bytes(ctx):
    prompt = [0, *ctx.randint(2, 64, 11).tolist()]
    for share_from in (1, 2):
        lm, cache = run(ctx, 7, prompt, share_from, 8)
        # Keys and values, float32, d_model numbers per token and layer.
        limit = 2 * share_from * 20 * lm.d_model * 4
        check.at_most("kv_bytes", kv_bytes(cache), limit, unit="bytes")


@case("a single-token prompt, then twelve decode steps")
def single_token_prompt(ctx):
    run(ctx, 9, [0], 2, 12)
    run(ctx, 9, [7], 3, 4)


@case("twelve decode steps on another model: every step reads the newest keys")
def many_steps(ctx):
    # Catches upper layers that read the cache from before the step, and a new token
    # embedded at position 0.
    run(ctx, 11, [0, 9, 30, 2, 51], 2, 12)
    run(ctx, 11, [0, 9, 30, 2, 51], 1, 6)


@case("six layers: share_from = 2, 3, and 5")
def six_layers(ctx):
    prompt = [0, *ctx.randint(2, 64, 9).tolist()]
    for share_from in (2, 3, 5):
        run(ctx, 4, prompt, share_from, 3, n_layers=6)


@case("two layers, share_from = 1: the smallest sharing model")
def two_layers(ctx):
    run(ctx, 6, [0, 33, 12, 60, 8, 21], 1, 5, n_layers=2)


@case("stored KV on a six-layer model with a long prompt")
def stored_bytes_six_layers(ctx):
    prompt = [0, *ctx.randint(2, 64, 39).tolist()]
    lm, cache = run(ctx, 13, prompt, 2, 4, n_layers=6)
    check.at_most("kv_bytes", kv_bytes(cache), 2 * 2 * 44 * lm.d_model * 4, unit="bytes")
