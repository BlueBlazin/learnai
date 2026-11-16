"""Tests for kv-cache-append.

Numeric cases compare one step with the oracle. Property cases feed a sequence in
chunks and compare with one full causal pass, computed here in numpy with the kit's
`ops.attention`, independently of the oracle.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import ops

FN = "attend_step"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def qkv(ctx, b, h, t, d, scale=1.0):
    return ctx.randn(b, h, t, d, scale=scale), ctx.randn(b, h, t, d), ctx.randn(b, h, t, d)


def full_pass(q, k, v):
    """Causal attention over the whole sequence at once: position i sees keys 0..i."""
    seq = q.shape[2]
    return ops.attention(q, k, v, np.tril(np.ones((seq, seq), dtype=bool)))


def feed_in_chunks(ctx, q, k, v, sizes):
    """Feed the sequence chunk by chunk; return the concatenated outputs and the last cache."""
    cache, outs, start = None, [], 0
    for size in sizes:
        stop = start + size
        result = ctx.call(FN, q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], cache)
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            f"{FN} must return a tuple (out, cache)",
        )
        out, cache = result
        check.that(
            isinstance(cache, tuple) and len(cache) == 2,
            f"{FN}: the returned cache must be a tuple (k, v)",
        )
        check.shape(out, q[:, :, start:stop].shape, label=f"out for tokens {start} to {stop - 1}")
        check.shape(cache[0], k[:, :, :stop].shape, label=f"cached keys after {stop} tokens")
        check.shape(cache[1], v[:, :, :stop].shape, label=f"cached values after {stop} tokens")
        outs.append(out)
        start = stop
    return np.concatenate(outs, axis=2), cache


def check_chunking(ctx, q, k, v, sizes):
    out, (ck, cv) = feed_in_chunks(ctx, q, k, v, sizes)
    name = "+".join(str(s) for s in sizes)
    check.allclose(out, full_pass(q, k, v), **TOL, label=f"outputs with chunks {name}")
    check.allclose(ck, k, atol=0.0, rtol=0.0, label=f"cached keys after chunks {name}")
    check.allclose(cv, v, atol=0.0, rtol=0.0, label=f"cached values after chunks {name}")


def grid(rows):
    return np.array(rows, dtype=np.float32)[None, None]


@case("the worked example: two tokens without a cache, then a third with it", public=True)
def worked_example(ctx):
    q = grid([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    k = grid([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = grid([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out, cache = ctx.check(FN, q[:, :, :2], k[:, :, :2], v[:, :, :2], None, **TOL)
    check.allclose(out, grid([[1.0, 2.0], [2.608860, 3.608860]]), atol=1e-5, label="out, step 1")
    out, cache = ctx.check(FN, q[:, :, 2:], k[:, :, 2:], v[:, :, 2:], cache, **TOL)
    check.allclose(out, grid([[3.510469, 4.510469]]), atol=1e-5, label="out, step 2")
    check.allclose(cache[0], k, atol=0.0, rtol=0.0, label="cached keys after step 2")
    check.allclose(cache[1], v, atol=0.0, rtol=0.0, label="cached values after step 2")


@case("no cache: three tokens at once is plain causal attention", public=True)
def prefill_without_cache(ctx):
    q, k, v = qkv(ctx, 2, 3, 3, 8)
    out, _ = ctx.check(FN, q, k, v, None, **TOL)
    check.allclose(out, full_pass(q, k, v), **TOL, label="out")


@case("a decode step: one new token over a cache of two", public=True)
def decode_step(ctx):
    q, k, v = qkv(ctx, 2, 3, 1, 8)
    cache = (ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8))
    ctx.check(FN, q, k, v, cache, **TOL)


# Rejects no_causal_mask_in_chunk, attends_to_cache_only, and mask_ignores_cache_offset.
@case("a chunk of two new tokens over a cache of two", public=True)
def chunk_over_cache(ctx):
    q, k, v = qkv(ctx, 2, 3, 2, 8)
    cache = (ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8))
    ctx.check(FN, q, k, v, cache, **TOL)


# The four ways of cutting three tokens into chunks.
CHUNKINGS = ([3], [1, 2], [2, 1], [1, 1, 1])


@case("prefill then decode: chunks 2+1 and 1+1+1 equal one full causal pass", public=True)
def prefill_then_decode(ctx):
    q, k, v = qkv(ctx, 2, 3, 3, 8)
    check_chunking(ctx, q, k, v, [2, 1])
    check_chunking(ctx, q, k, v, [1, 1, 1])


@case("every way of cutting three tokens into chunks equals one full causal pass")
def chunked_equals_full(ctx):
    def prop(rng):
        q, k, v = qkv(ctx, 2, 3, 3, 8, scale=float(rng.uniform(0.5, 3.0)))
        for sizes in CHUNKINGS:
            check_chunking(ctx, q, k, v, sizes)

    check.forall(prop, trials=4)


# Rejects returns_stale_cache and prepends_new_entries by the cache alone.
@case("the returned cache is the old entries followed by the new ones")
def cache_contents(ctx):
    q, k, v = qkv(ctx, 2, 3, 2, 8)
    old_k, old_v = ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8)
    _, (new_k, new_v) = ctx.call(FN, q, k, v, (old_k, old_v))
    check.shape(new_k, (2, 3, 4, 8), label="cached keys")
    check.shape(new_v, (2, 3, 4, 8), label="cached values")
    check.allclose(new_k, np.concatenate((old_k, k), axis=2), atol=0.0, rtol=0.0,
                   label="cached keys")  # fmt: skip
    check.allclose(new_v, np.concatenate((old_v, v), axis=2), atol=0.0, rtol=0.0,
                   label="cached values")  # fmt: skip


# Rejects appends_on_head_axis where it does not crash: S = T, so both axes fit.
@case("one head and one batch row, D = 16, a cache as long as the chunk")
def single_head(ctx):
    q, k, v = qkv(ctx, 1, 1, 2, 16)
    cache = (ctx.randn(1, 1, 2, 16), ctx.randn(1, 1, 2, 16))
    ctx.check(FN, q, k, v, cache, **TOL)


# B = H = S = T = D: every axis has the same length.
@case("all axes the same length: B = H = S = T = D = 4")
def all_axes_equal(ctx):
    q, k, v = qkv(ctx, 4, 4, 4, 4)
    cache = (ctx.randn(4, 4, 4, 4), ctx.randn(4, 4, 4, 4))
    ctx.check(FN, q, k, v, cache, **TOL)


@case("changing the last token of a chunk never changes the outputs before it")
def causality_within_chunk(ctx):
    def prop(rng):
        q, k, v = qkv(ctx, 2, 3, 2, 8)
        cache = (ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8))
        k2, v2 = k.copy(), v.copy()
        k2[:, :, 1:] = ctx.randn(2, 3, 1, 8, scale=3.0)
        v2[:, :, 1:] = ctx.randn(2, 3, 1, 8, scale=3.0)
        before, _ = ctx.call(FN, q, k, v, cache)
        after, _ = ctx.call(FN, q, k2, v2, cache)
        check.finite(before, label="out")
        check.allclose(
            after[:, :, :1],
            before[:, :, :1],
            atol=1e-5,
            label="output of new token 0 after changing new token 1",
        )

    check.forall(prop, trials=4)


@case("large scores: nearly one-hot attention over cache and chunk")
def sharp_attention(ctx):
    q, k, v = qkv(ctx, 2, 3, 2, 8, scale=8.0)
    cache = (ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8))
    out, _ = ctx.check(FN, q, k, v, cache, **TOL)
    check.finite(out, label="out")
