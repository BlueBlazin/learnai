"""Tests for preallocated-kv-cache.

The tests play the decoder: they write chunks of keys and values and attend after each
write. Outputs are compared with the oracle and with one full causal pass over the
tokens written so far, computed in numpy with the kit's `ops.attention`. Several cases
start from a buffer whose unused tail holds NaN, infinities, or large values, which a
correct solution never reads.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import ops

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
B, H, C, D = 2, 3, 8, 8  # the sizes most cases share (fewer shapes: faster in JAX)


def tokens(ctx, n, b=B, h=H, d=D, scale=1.0):
    """Queries, keys, and values of n tokens."""
    return ctx.randn(b, h, n, d, scale=scale), ctx.randn(b, h, n, d), ctx.randn(b, h, n, d)


def full_pass(q, k, v):
    """Causal attention over the whole sequence at once: position i sees keys 0..i."""
    seq = q.shape[2]
    return ops.attention(q, k, v, np.tril(np.ones((seq, seq), dtype=bool)))


def dirty_cache(fill, length=0, b=B, h=H, c=C, d=D, k=None, v=None):
    """A cache built by the test: `length` real tokens, and `fill` in every unused slot."""
    k_buf = np.full((b, h, c, d), fill, dtype=np.float32)
    v_buf = np.full((b, h, c, d), fill, dtype=np.float32)
    if length:
        k_buf[:, :, :length], v_buf[:, :, :length] = k[:, :, :length], v[:, :, :length]
    return k_buf, v_buf, length


def check_cache(cache, shape, k, v, length, what):
    """The structure of a cache, its length, and the slots that hold tokens."""
    check.that(
        isinstance(cache, tuple) and len(cache) == 3,
        f"{what}: a cache must be a tuple (k, v, length)",
    )
    k_buf, v_buf, got_length = cache
    check.that(
        type(got_length) is int,
        f"{what}: the cache length must be a Python int",
        f"got {type(got_length).__name__}",
    )
    check.shape(k_buf, shape, label=f"{what}: key buffer")
    check.shape(v_buf, shape, label=f"{what}: value buffer")
    check.equal(got_length, length, label=f"{what}: cache length")
    if not length:
        return
    check.allclose(k_buf[:, :, :length], k[:, :, :length], atol=0.0, rtol=0.0,
                   label=f"{what}: keys in slots 0 to {length - 1}")  # fmt: skip
    check.allclose(v_buf[:, :, :length], v[:, :, :length], atol=0.0, rtol=0.0,
                   label=f"{what}: values in slots 0 to {length - 1}")  # fmt: skip


def write_and_attend(ctx, cache, q, k, v, sizes, start=0):
    """Write the tokens from `start` on in chunks, attending after every write."""
    want = full_pass(q, k, v)
    shape = cache[0].shape
    done = start
    for size in sizes:
        stop = done + size
        cache = ctx.call("cache_write", cache, k[:, :, done:stop], v[:, :, done:stop])
        what = f"after writing tokens {done} to {stop - 1}"
        check_cache(cache, shape, k, v, stop, what)
        out = ctx.call("cache_attend", q[:, :, done:stop], cache)
        check.shape(out, q[:, :, done:stop].shape, label=f"cache_attend {what}")
        check.allclose(out, want[:, :, done:stop], **TOL, label=f"cache_attend {what}")
        done = stop
    return cache


def grid(rows):
    return np.array(rows, dtype=np.float32)[None, None]


@case("the worked example: capacity 4, two tokens written, then a third", public=True)
def worked_example(ctx):
    q = grid([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    k = grid([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = grid([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cache = ctx.check("cache_init", 4, 1, 1, 2)
    cache = ctx.check("cache_write", cache, k[:, :, :2], v[:, :, :2])
    check_cache(cache, (1, 1, 4, 2), k, v, 2, "after the first write")
    out = ctx.check("cache_attend", q[:, :, :2], cache, **TOL)
    check.allclose(out, grid([[1.0, 2.0], [2.608860, 3.608860]]), atol=1e-5, label="first output")
    cache = ctx.check("cache_write", cache, k[:, :, 2:], v[:, :, 2:])
    check_cache(cache, (1, 1, 4, 2), k, v, 3, "after the second write")
    out = ctx.check("cache_attend", q[:, :, 2:], cache, **TOL)
    check.allclose(out, grid([[3.510469, 4.510469]]), atol=1e-5, label="second output")


@case("cache_init: zero buffers of shape (B, H, capacity, D) and length 0", public=True)
def init_shapes(ctx):
    cache = ctx.check("cache_init", C, B, H, D)
    check_cache(
        cache, (B, H, C, D), np.zeros((B, H, C, D)), np.zeros((B, H, C, D)), 0, "cache_init"
    )
    check.allclose(cache[0], np.zeros((B, H, C, D)), atol=0.0, rtol=0.0, label="key buffer")
    check.allclose(cache[1], np.zeros((B, H, C, D)), atol=0.0, rtol=0.0, label="value buffer")
    # capacity comes first: a (5, 1, 2, 4) call has 5 slots, not 5 batch rows.
    check_cache(ctx.call("cache_init", 5, 1, 2, 4), (1, 2, 5, 4), None, None, 0, "cache_init")


# Rejects concatenates (the shape), attends_whole_buffer (the zero tail takes weight),
# and no_causal_mask_in_chunk (three queries after a three-token write).
@case("prefill three tokens, then decode two: shapes stay fixed, outputs match", public=True)
def prefill_then_decode(ctx):
    q, k, v = tokens(ctx, 5)
    cache = ctx.call("cache_init", C, B, H, D)
    write_and_attend(ctx, cache, q, k, v, [3, 1, 1])


# Rejects masks_scores_only: a zero weight does not neutralize a NaN value.
@case("the unused tail holds NaN: it must not reach the output", public=True)
def nan_tail(ctx):
    q, k, v = tokens(ctx, 5)
    cache = dirty_cache(np.nan, 3, k=k, v=v)
    out = ctx.check("cache_attend", q[:, :, :3], cache, **TOL)
    check.finite(out, label="cache_attend over a buffer with a NaN tail")
    write_and_attend(ctx, cache, q, k, v, [1, 1], start=3)


# Rejects capacity_check_off_by_one and truncates_on_overflow.
@case("filling the buffer exactly is fine; one token more raises ValueError", public=True)
def overflow(ctx):
    q, k, v = tokens(ctx, 6)
    cache = ctx.call("cache_init", 5, B, H, D)
    cache = write_and_attend(ctx, cache, q, k, v, [3, 1, 1])
    check.raises(
        ValueError, ctx.call, "cache_write", cache, k[:, :, 5:], v[:, :, 5:],
        label="cache_write into a full buffer",
    )  # fmt: skip


# Rejects length_off_by_one where a single chunk would hide it.
@case("two multi-token writes in a row: three tokens, then two")
def two_chunks(ctx):
    q, k, v = tokens(ctx, 5)
    cache = dirty_cache(1e30)
    write_and_attend(ctx, cache, q, k, v, [3, 2])


@case("a chunk that would cross the end of the buffer raises ValueError")
def overflow_by_chunk(ctx):
    q, k, v = tokens(ctx, 6)
    cache = dirty_cache(0.0, 3, c=5, k=k, v=v)
    check.raises(
        ValueError, ctx.call, "cache_write", cache, k[:, :, 3:], v[:, :, 3:],
        label="cache_write of 3 tokens with 2 free slots",
    )  # fmt: skip
    # Exactly two more still fit.
    write_and_attend(ctx, cache, q[:, :, :5], k[:, :, :5], v[:, :, :5], [2], start=3)


@case("a tail of infinities, and a tail of stale entries from an earlier sequence")
def garbage_tails(ctx):
    q, k, v = tokens(ctx, 5)
    for fill in (np.inf, -np.inf):
        cache = dirty_cache(fill, 3, k=k, v=v)
        out = ctx.check("cache_attend", q[:, :, :3], cache, **TOL)
        check.finite(out, label="cache_attend over a buffer with an infinite tail")
    # A reused buffer: every slot holds a plausible old token.
    stale_k, stale_v = ctx.randn(B, H, C, D, scale=3.0), ctx.randn(B, H, C, D, scale=3.0)
    write_and_attend(ctx, (stale_k, stale_v, 0), q, k, v, [3, 1, 1])


@case("any chunking over a dirty buffer equals one full causal pass")
def chunked_equals_full(ctx):
    def prop(rng):
        q, k, v = tokens(ctx, 5, scale=float(rng.uniform(0.5, 3.0)))
        fill = [np.nan, 1e30, -1e30, np.inf][int(rng.integers(0, 4))]
        sizes = [[3, 1, 1], [3, 2], [1, 1, 1, 1, 1], [1, 2, 2]][int(rng.integers(0, 4))]
        write_and_attend(ctx, dirty_cache(fill), q, k, v, sizes)

    check.forall(prop, trials=5)


@case("decode into the last slot of a buffer whose tail is NaN")
def decode_until_full(ctx):
    q, k, v = tokens(ctx, 5)
    cache = dirty_cache(np.nan, 3, c=5, k=k, v=v)
    write_and_attend(ctx, cache, q, k, v, [1, 1], start=3)


@case("capacity 1: one token fits, the second raises ValueError")
def capacity_one(ctx):
    q, k, v = tokens(ctx, 2, b=1, h=1, d=4)
    cache = ctx.call("cache_init", 1, 1, 1, 4)
    cache = write_and_attend(ctx, cache, q, k, v, [1])
    check.raises(
        ValueError, ctx.call, "cache_write", cache, k[:, :, 1:], v[:, :, 1:],
        label="cache_write into a full buffer of capacity 1",
    )  # fmt: skip
