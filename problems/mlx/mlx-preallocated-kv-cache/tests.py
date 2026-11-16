"""Tests for mlx-preallocated-kv-cache.

Every case feeds a cache a sequence of chunks of keys and values and, after each
update, evaluates what came back (as attention would) and checks, in this order:

1. the result: the returned keys and values, exactly, against the concatenation of
   everything written so far (numpy), with the dtype of what was written;
2. the bookkeeping: `offset` is a Python int with the right value, and the buffers
   `cache.keys` and `cache.values` have exactly the capacity the growth rule gives.
   The number of times the buffers changed shape is reported as `reallocations`: an
   exact count;
3. evaluation: `EvalSpy` watches the solution file during every update.

One case, on the GPU, measures what the growth rule is for: the memory MLX's
allocator holds (in use plus kept for reuse) after 600 single-token updates.
"""

import math

import mlx.core as mx
import numpy as np

from learnai.judge import case, check
from learnai.kit import EvalSpy, mlx_memory

FN = "update_and_fetch"
MIB = 2**20


def make_chunks(ctx, lengths, *, batch=1, heads=2, key_dim=4, value_dim=None, dtype=np.float32):
    """Random (keys, values) chunks of the given token counts."""
    value_dim = key_dim if value_dim is None else value_dim
    return [
        (
            ctx.randn(batch, heads, length, key_dim).astype(dtype),
            ctx.randn(batch, heads, length, value_dim).astype(dtype),
        )
        for length in lengths
    ]


def capacity_after(capacity, offset, new, step):
    """The growth rule of the statement."""
    if capacity is None or offset + new > capacity:
        return offset + math.ceil(new / step) * step
    return capacity


def check_fetched(got, want, name, call):
    check.that(
        isinstance(got, mx.array),
        f"{FN}: must return a tuple (keys, values) of mx.arrays",
        f"{name} is a {type(got).__name__}",
    )
    check.that(
        tuple(got.shape) == want.shape,
        f"{FN}: returned {name} of shape {tuple(got.shape)}, which is not the shape of the "
        "tokens stored so far",
        f"expected {want.shape} after update {call}",
    )
    check.that(
        np.dtype(str(got.dtype).rsplit(".", 1)[-1]) == want.dtype,
        f"{FN}: returned {name} of dtype {got.dtype}, which is not the dtype that was written",
        f"expected {want.dtype} after update {call}",
    )
    check.allclose(
        np.array(got), want, rtol=0.0, atol=0.0, label=f"{name} returned by update {call}"
    )


def feed(ctx, chunks, *, step=None, every=1):
    """Run the solution on `chunks`; check results, bookkeeping, and evaluation.

    `every`: compare the full contents on every `every`-th update (and the last); the
    shapes, the offset, and the capacity are checked on all of them.
    """
    cache = ctx.solution.KVCache() if step is None else ctx.solution.KVCache(step=step)
    step = 256 if step is None else step
    check.that(
        cache.keys is None and cache.values is None and cache.offset == 0,
        "KVCache: a new cache must have keys = None, values = None, and offset = 0",
    )
    all_keys = np.concatenate([k for k, _ in chunks], axis=2)
    all_values = np.concatenate([v for _, v in chunks], axis=2)
    capacity, offset, growths, reallocations = None, 0, 0, 0
    shape_before = None
    with EvalSpy() as spy:
        for call, (keys, values) in enumerate(chunks, start=1):
            result = cache.update_and_fetch(mx.array(keys), mx.array(values))
            check.that(
                isinstance(result, tuple) and len(result) == 2,
                f"{FN}: must return a tuple (keys, values)",
                f"got {type(result).__name__}",
            )
            mx.eval(result)  # what attention does with the result
            new = keys.shape[2]
            grown = capacity_after(capacity, offset, new, step)
            growths += grown != capacity
            capacity, offset = grown, offset + new

            # 1. The result.
            if call % every == 0 or call == len(chunks):
                check_fetched(result[0], all_keys[:, :, :offset], "keys", call)
                check_fetched(result[1], all_values[:, :, :offset], "values", call)
            else:
                check.that(
                    result[0].shape[2] == offset and result[1].shape[2] == offset,
                    f"{FN}: returned {result[0].shape[2]} tokens, which is not the number stored",
                    f"expected {offset} after update {call}",
                )

            # 2. The bookkeeping.
            check.that(
                type(cache.offset) is int and cache.offset == offset,
                f"{FN}: offset is {cache.offset!r} after an update, which is not the number "
                "of tokens stored",
                f"expected the int {offset} after update {call}",
            )
            check.that(
                isinstance(cache.keys, mx.array) and isinstance(cache.values, mx.array),
                f"{FN}: cache.keys and cache.values must be the buffers (mx.arrays)",
            )
            shape = (tuple(cache.keys.shape), tuple(cache.values.shape))
            reallocations += shape != shape_before
            shape_before = shape
            expected = (
                (*keys.shape[:2], capacity, keys.shape[3]),
                (*values.shape[:2], capacity, values.shape[3]),
            )
            check.that(
                shape == expected,
                f"{FN}: the buffers have {shape[0][2]} slots after an update, which is not "
                "the capacity the growth rule gives",
                f"update {call}: expected keys {expected[0]} and values {expected[1]}, "
                f"got {shape[0]} and {shape[1]}",
            )
    check.at_most("reallocations", reallocations, growths)

    # The buffers themselves hold the tokens in their first `offset` slots.
    check.allclose(
        np.array(cache.keys[:, :, :offset]), all_keys, rtol=0.0, atol=0.0, label="cache.keys"
    )
    check.allclose(
        np.array(cache.values[:, :, :offset]), all_values, rtol=0.0, atol=0.0, label="cache.values"
    )

    # 3. Evaluation.
    seen = spy.evals + spy.async_evals + spy.implicit
    check.that(
        seen == 0,
        f"{FN}: evaluated {seen} time(s) (mx.eval, .item(), np.array, ...); an update must "
        "only build a graph",
        spy.describe(),
    )
    return cache


@case("the worked example: step 4, chunks of 3, 1, 1, and 6 tokens", public=True)
def worked_example(ctx):
    numbers = np.arange(1, 12, dtype=np.float32).reshape(1, 1, 11, 1)
    chunks = [
        (numbers[:, :, a:b], 10 * numbers[:, :, a:b]) for a, b in [(0, 3), (3, 4), (4, 5), (5, 11)]
    ]
    cache = feed(ctx, chunks, step=4)
    check.equal(cache.keys.shape[2], 13, label="capacity after the four updates")


@case("a 5-token prompt, then 300 tokens one at a time: 2 reallocations", public=True)
def prompt_then_tokens(ctx):
    # Rejects concat_every_step and exact_growth (a new shape on every update) and
    # returns_whole_buffer (256 tokens returned after the first update).
    feed(ctx, make_chunks(ctx, [5] + [1] * 300), every=25)


@case("a 300-token prompt needs two steps at once; then tokens up to 520", public=True)
def long_prompt(ctx):
    # Rejects grows_one_step: 300 tokens do not fit in 256 slots.
    feed(ctx, make_chunks(ctx, [300] + [1] * 220), every=20)


@case("float16 keys and values stay float16, in the buffers and in the result", public=True)
def half_precision(ctx):
    # Rejects float32_buffers.
    cache = feed(ctx, make_chunks(ctx, [3, 1, 1, 6, 1], dtype=np.float16), step=4)
    check.that(
        cache.keys.dtype == mx.float16 and cache.values.dtype == mx.float16,
        f"{FN}: the buffers are {cache.keys.dtype} and {cache.values.dtype} for float16 input",
        "allocate the buffers with the dtype of the arrays written to them",
    )


# Hidden cases.


@case("chunks that do not fit while the buffer is partly full")
def growth_from_a_partial_buffer(ctx):
    # Rejects grows_from_capacity: the unused tail is dropped when the buffer grows.
    feed(ctx, make_chunks(ctx, [3, 3, 9, 1, 2, 7, 4]), step=4)
    feed(ctx, make_chunks(ctx, [300, 300, 1, 211, 1]), every=1)


@case("exact fits: chunks that end on a step boundary, and a chunk of exactly one step")
def exact_fits(ctx):
    feed(ctx, make_chunks(ctx, [4, 4, 1, 3, 8, 1]), step=4)
    feed(ctx, make_chunks(ctx, [256, 1, 255, 1]))


@case("old tokens survive every reallocation: 40 updates with step 4")
def survives_growth(ctx):
    # Rejects drops_old_on_growth.
    feed(ctx, make_chunks(ctx, [1] * 40), step=4)


@case("a batch of 3, 2 heads, and values wider than keys")
def batch_and_dims(ctx):
    # Rejects value_dim_from_keys.
    feed(ctx, make_chunks(ctx, [7, 1, 1, 9, 1], batch=3, key_dim=4, value_dim=6), step=8)


@case("two caches are independent, and step 1 reallocates on every token")
def two_caches(ctx):
    first = make_chunks(ctx, [2, 1, 1])
    second = make_chunks(ctx, [1, 3])
    a = feed(ctx, first, step=4)
    b = feed(ctx, second, step=1)
    check.that(
        a.offset == 4 and b.offset == 4 and a.keys.shape[2] == 4 and b.keys.shape[2] == 4,
        "KVCache: two caches share state",
        "each instance has its own buffers, offset, and step",
    )
    check.allclose(
        np.array(a.keys[:, :, :4]),
        np.concatenate([k for k, _ in first], axis=2),
        rtol=0.0,
        atol=0.0,
        label="the first cache, after the second was used",
    )


@case("updates that are never evaluated one by one still give the right cache at the end")
def lazy_chain(ctx):
    chunks = make_chunks(ctx, [6] + [1] * 30)
    cache = ctx.solution.KVCache(step=8)
    for keys, values in chunks:
        fetched = cache.update_and_fetch(mx.array(keys), mx.array(values))
    check_fetched(fetched[0], np.concatenate([k for k, _ in chunks], axis=2), "keys", "31")
    check_fetched(fetched[1], np.concatenate([v for _, v in chunks], axis=2), "values", "31")
    check.equal(cache.keys.shape[2], 40, label="capacity")


@case(
    "600 single-token updates: the allocator holds at most 24 MiB",
    requires="mlx-gpu",
)
def allocator_memory(ctx):
    # Rejects concat_every_step and exact_growth by what they cost: 600 buffers of 600
    # different sizes, none of which MLX can reuse for the next one (about 70 MiB
    # here; the reference holds 9 MiB).
    chunks = make_chunks(ctx, [1] * 600, heads=4, key_dim=64)  # 1 KiB per token and buffer
    arrays = [(mx.array(k), mx.array(v)) for k, v in chunks]
    mx.eval(arrays)
    with mlx_memory() as mem:
        cache = ctx.solution.KVCache()
        for keys, values in arrays:
            fetched = cache.update_and_fetch(keys, values)
            mx.eval(fetched)
    check_fetched(fetched[0], np.concatenate([k for k, _ in chunks], axis=2), "keys", "600")
    held = (mem.active_bytes + mem.cache_bytes) / MIB
    ctx.metric("active_memory", round(mem.active_mib, 2), "MiB")
    check.at_most("allocator_memory", round(held, 2), 24.0, unit="MiB")
