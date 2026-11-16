"""Tests for sliding-window-cache.

Each case writes a stream of tokens in chunks and checks, after every write, the three
things the statement specifies: the returned keys and values (the first `keep` tokens,
then the most recent `max_size - keep`, oldest first), the layout of the buffer (every
cached token in its own slot), and the bookkeeping (buffer shapes, the offset). The
expected contents are computed here from the token numbers, independently of the
oracle. Entries are copies, so comparisons are exact.
"""

import numpy as np

from learnai.judge import case, check

FN = "rotating_write"
B, H, D = 2, 2, 4  # the sizes most cases share (fewer shapes: faster in JAX)
EXACT = {"rtol": 0.0, "atol": 0.0}


def live_tokens(end, max_size, keep):
    """Numbers of the tokens cached after `end` tokens were written, oldest first."""
    return [*range(min(keep, end)), *range(max(keep, end - (max_size - keep)), end)]


def slot_of(n, max_size, keep):
    return n if n < keep else keep + (n - keep) % (max_size - keep)


def empty_cache(max_size, fill=np.nan, b=B, h=H, d=D):
    """An empty cache whose slots hold `fill`: nothing a correct solution ever returns."""
    shape = (b, h, max_size, d)
    return np.full(shape, fill, dtype=np.float32), np.full(shape, fill, dtype=np.float32), 0


def write_stream(ctx, k, v, sizes, max_size, keep, cache=None):
    """Write the tokens of k and v in chunks of the given sizes, checking after each."""
    b, h, _, d = k.shape
    cache = empty_cache(max_size, b=b, h=h, d=d) if cache is None else cache
    done = cache[2]
    for size in sizes:
        end = done + size
        result = ctx.call(FN, cache, k[:, :, done:end], v[:, :, done:end], max_size, keep)
        what = f"after {end} tokens (last write: {size})"
        check.that(
            isinstance(result, tuple) and len(result) == 3,
            f"{FN} must return a tuple (cache, keys, values)",
        )
        cache, keys, values = result
        check.that(
            isinstance(cache, tuple) and len(cache) == 3,
            f"{FN}: the returned cache must be a tuple (k, v, offset)",
        )
        check.that(
            type(cache[2]) is int,
            f"{FN}: the cache offset must be a Python int",
            f"got {type(cache[2]).__name__}",
        )
        check.equal(cache[2], end, label=f"offset {what}")
        check.shape(cache[0], (b, h, max_size, d), label=f"key buffer {what}")
        check.shape(cache[1], (b, h, max_size, d), label=f"value buffer {what}")
        # Contents: the sinks, then the most recent tokens, in temporal order.
        live = live_tokens(end, max_size, keep)
        check.allclose(keys, k[:, :, live], **EXACT, label=f"keys in temporal order {what}")
        check.allclose(values, v[:, :, live], **EXACT, label=f"values in temporal order {what}")
        # Layout: each cached token sits in its own slot of the ring.
        slots = [slot_of(n, max_size, keep) for n in live]
        check.allclose(cache[0][:, :, slots], k[:, :, live], **EXACT,
                       label=f"key buffer layout {what}")  # fmt: skip
        check.allclose(cache[1][:, :, slots], v[:, :, live], **EXACT,
                       label=f"value buffer layout {what}")  # fmt: skip
        done = end
    return cache


def stream(ctx, n, b=B, h=H, d=D):
    return ctx.randn(b, h, n, d), ctx.randn(b, h, n, d)


def numbered(first, last):
    """Tokens first .. last - 1 as keys of shape (1, 1, T, 1): the key of token n is n."""
    return np.arange(first, last, dtype=np.float32).reshape(1, 1, -1, 1)


@case("the worked example: max_size 5, keep 2, writes of 4, 2, 2, and 1 tokens", public=True)
def worked_example(ctx):
    cache = (np.zeros((1, 1, 5, 1), np.float32), np.zeros((1, 1, 5, 1), np.float32), 0)
    steps = [
        ((0, 4), [0, 1, 2, 3, 0], [0, 1, 2, 3]),
        ((4, 6), [0, 1, 5, 3, 4], [0, 1, 3, 4, 5]),
        ((6, 8), [0, 1, 5, 6, 7], [0, 1, 5, 6, 7]),
        ((8, 9), [0, 1, 8, 6, 7], [0, 1, 6, 7, 8]),
    ]
    for (first, last), buffer, order in steps:
        k_new = numbered(first, last)
        cache, keys, values = ctx.check(FN, cache, k_new, 100 + k_new, 5, 2)
        what = f"after writing tokens {first} to {last - 1}"
        check.equal(cache[2], last, label=f"offset {what}")
        check.allclose(cache[0].ravel(), np.array(buffer, np.float32), **EXACT,
                       label=f"key buffer {what}")  # fmt: skip
        check.allclose(keys.ravel(), np.array(order, np.float32), **EXACT, label=f"keys {what}")
        check.allclose(values.ravel(), 100 + np.array(order, np.float32), **EXACT,
                       label=f"values {what}")  # fmt: skip


@case("before the buffer is full it is a plain preallocated cache", public=True)
def not_yet_full(ctx):
    k, v = stream(ctx, 6)
    write_stream(ctx, k, v, [3, 1, 1, 1], max_size=6, keep=2)


# Rejects returns_slot_order and rolls_sinks_into_order: ten single-token writes wrap
# the ring of four slots, and the oldest entry is in a different slot each time.
@case("one token at a time through a wrap: max_size 6, keep 2", public=True)
def token_by_token(ctx):
    k, v = stream(ctx, 12)
    cache = write_stream(ctx, k, v, [3], max_size=6, keep=2)
    write_stream(ctx, k, v, [1] * 9, max_size=6, keep=2, cache=cache)


# Rejects evicts_sinks even for a solution that is right with keep = 0.
@case("the first keep tokens survive two full turns of the ring", public=True)
def sinks_survive(ctx):
    k, v = stream(ctx, 15)
    cache = write_stream(ctx, k, v, [3] * 5, max_size=6, keep=2)
    check.allclose(cache[0][:, :, :2], k[:, :, :2], **EXACT, label="keys in the two sink slots")
    check.allclose(cache[1][:, :, :2], v[:, :, :2], **EXACT, label="values in the two sink slots")


# Rejects restarts_at_wrap: with 5 tokens cached, a 3-token write fills slot 5 and
# continues in slots 2 and 3.
@case("a multi-token write that crosses the wrap point", public=True)
def crosses_wrap(ctx):
    k, v = stream(ctx, 11)
    write_stream(ctx, k, v, [3, 2, 3, 3], max_size=6, keep=2)


@case("keep = 0: a pure ring over the whole buffer")
def no_sinks(ctx):
    k, v = stream(ctx, 12)
    write_stream(ctx, k, v, [3, 1, 3, 2, 3], max_size=6, keep=0)


@case("a write longer than the window: only its last tokens are kept")
def chunk_longer_than_window(ctx):
    k, v = stream(ctx, 15)
    # 3 tokens, then 6 at once into a ring of 4, then 6 more.
    write_stream(ctx, k, v, [3, 6, 6], max_size=6, keep=2)


@case("a first write longer than the buffer, and one that covers the sinks only in part")
def long_first_write(ctx):
    k, v = stream(ctx, 12)
    write_stream(ctx, k, v, [9, 3], max_size=6, keep=2)
    # One token, then a chunk that finishes the sinks and starts the ring.
    write_stream(ctx, k, v, [1, 3, 2, 6], max_size=6, keep=2)


@case("a ring of one slot: max_size 6, keep 5")
def ring_of_one(ctx):
    k, v = stream(ctx, 9)
    write_stream(ctx, k, v, [6, 1, 1, 1], max_size=6, keep=5)


@case("exactly full, then exactly one turn of the ring later")
def boundaries(ctx):
    k, v = stream(ctx, 12)
    write_stream(ctx, k, v, [6, 2, 2, 2], max_size=6, keep=2)


@case("random chunk sizes: the cache always holds the sinks and the most recent tokens")
def contents_property(ctx):
    def prop(rng):
        keep = int(rng.choice([0, 2, 5]))
        sizes = [int(s) for s in rng.choice([1, 2, 3, 6], size=6)]
        k, v = stream(ctx, sum(sizes))
        write_stream(ctx, k, v, sizes, max_size=6, keep=keep)

    check.forall(prop, trials=6)


@case("other sizes: B = 1, H = 3, D = 8, max_size 4, keep 1")
def other_sizes(ctx):
    k, v = stream(ctx, 9, b=1, h=3, d=8)
    write_stream(ctx, k, v, [2, 1, 3, 1, 2], max_size=4, keep=1)
