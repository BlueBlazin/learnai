"""Tests for cached-rope-offsets.

`rope_positions` is compared exactly. `rotate_new` is compared with the oracle and, in
the chunked cases, with the kit's `ops.rope` applied once to the whole sequence in
numpy. Angles are positions times frequencies in float32, so cases with positions below
64 use a tight tolerance and the case with large positions a looser one (see the
statement's Constraints).
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import ops

TIGHT = {"rtol": 1e-4, "atol": 1e-4}  # positions below 64
LOOSE = {"rtol": 1e-3, "atol": 5e-3}  # positions below 2048


def lengths(*values):
    return np.array(values, dtype=np.int64)


def positions_for(ctx, cache_lengths, n_new):
    """Call rope_positions and insist on the specified shape and values."""
    got = ctx.call("rope_positions", cache_lengths, n_new)
    want = cache_lengths[:, None] + np.arange(n_new)
    check.shape(got, want.shape, label="rope_positions")
    check.that(
        np.asarray(got).dtype.kind in "iu",
        "rope_positions must return an integer tensor",
        f"got dtype {np.asarray(got).dtype}",
    )
    check.equal(got, want, label="rope_positions")
    return got


def check_chunking(ctx, q, k, start, sizes, theta, tol):
    """Rotate a sequence chunk by chunk, as a cached model does, and compare with one pass.

    The test plays the cache: it tracks each row's length, and keeps the rotated keys
    of earlier chunks exactly as they were returned.
    """
    cached, rotated_q, done = start.copy(), [], 0
    cached_k = []
    for size in sizes:
        positions = positions_for(ctx, cached, size)
        chunk = slice(done, done + size)
        q_rot, k_rot = ctx.call("rotate_new", q[:, :, chunk], k[:, :, chunk], positions, theta)
        check.shape(q_rot, q[:, :, chunk].shape, label="rotated queries")
        check.shape(k_rot, k[:, :, chunk].shape, label="rotated keys")
        rotated_q.append(q_rot)
        cached_k.append(k_rot)  # stored rotated, never touched again
        cached = cached + size
        done += size
    name = "+".join(str(s) for s in sizes)
    full = start[:, None] + np.arange(q.shape[2])
    check.allclose(
        np.concatenate(rotated_q, axis=2),
        ops.rope(q, full, theta),
        **tol,
        label=f"rotated queries with chunks {name}",
    )
    check.allclose(
        np.concatenate(cached_k, axis=2),
        ops.rope(k, full, theta),
        **tol,
        label=f"cached keys with chunks {name}",
    )


@case("the worked example: cache lengths [3, 0], two new tokens, theta = 100", public=True)
def worked_example(ctx):
    positions = positions_for(ctx, lengths(3, 0), 2)
    x = np.tile(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32), (2, 1, 2, 1))
    q_rot, k_rot = ctx.check("rotate_new", x, x, positions, 100.0, **TIGHT)
    expected = np.array(
        [
            [[[-0.989992, 0.141120, 0.955337, 0.295520], [-0.653644, -0.756802, 0.921061, 0.389418]]],
            [[[1.0, 0.0, 1.0, 0.0], [0.540302, 0.841471, 0.995004, 0.099833]]],
        ],
        dtype=np.float32,
    )  # fmt: skip
    check.allclose(q_rot, expected, atol=1e-5, label="rotated queries")
    check.allclose(k_rot, expected, atol=1e-5, label="rotated keys")


# Rejects positions_restart_at_zero, one_offset_for_batch, and positions_off_by_one.
@case("positions for three rows with cache lengths [0, 5, 17]", public=True)
def positions_per_row(ctx):
    positions_for(ctx, lengths(0, 5, 17), 1)
    positions_for(ctx, lengths(0, 5, 17), 4)


# Rejects keys_not_rotated and first_row_positions. B = T = 2 hides a transposed table.
@case("rotating two new tokens in two rows at different positions", public=True)
def rotate_two_rows(ctx):
    positions = lengths(9, 2)[:, None] + np.arange(2)
    ctx.check(
        "rotate_new", ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8), positions, 10000.0, **TIGHT
    )


@case("prefill then decode equals rotating the whole sequence at once", public=True)
def chunked_equals_full(ctx):
    q, k = ctx.randn(2, 3, 4, 8), ctx.randn(2, 3, 4, 8)
    start = lengths(0, 6)  # row 1 continues a conversation that is already cached
    check_chunking(ctx, q, k, start, [4], 10000.0, TIGHT)
    check_chunking(ctx, q, k, start, [2, 1, 1], 10000.0, TIGHT)


@case("any chunking, any starting lengths: the rotated tensors do not depend on the split")
def chunking_property(ctx):
    def prop(rng):
        q, k = ctx.randn(2, 3, 4, 8), ctx.randn(2, 3, 4, 8)
        start = rng.integers(0, 40, size=2)
        sizes = [[2, 2], [1, 1, 2], [1, 2, 1], [1, 1, 1, 1], [2, 1, 1]][int(rng.integers(0, 5))]
        check_chunking(ctx, q, k, start, sizes, float(rng.choice([100.0, 500.0, 10000.0])), TIGHT)

    check.forall(prop, trials=5)


# The longest cache is not in row 0, and not every row has one.
@case("rows in any order of length, a batch of one, and a single new token")
def row_orders(ctx):
    positions_for(ctx, lengths(4, 31, 0, 12), 2)
    positions_for(ctx, lengths(7, 7), 2)
    positions_for(ctx, lengths(23), 1)
    positions_for(ctx, lengths(0), 4)


# Rejects ignores_theta without help from the worked example.
@case("a different base: theta = 500")
def other_theta(ctx):
    positions = lengths(3, 20)[:, None] + np.arange(2)
    ctx.check("rotate_new", ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8), positions, 500.0, **TIGHT)
    ctx.check("rotate_new", ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8), positions, 30.0, **TIGHT)


@case("a new query scores a cached key by their distance, whatever the cache length")
def scores_depend_on_distance(ctx):
    def prop(rng):
        q, k = ctx.randn(2, 3, 2, 8), ctx.randn(2, 3, 2, 8)
        base_len = rng.integers(0, 20, size=2)
        shift = rng.integers(1, 30, size=2)  # a different shift per row

        def scores(cache_lengths):
            # Token 0 goes in first and its key is cached; token 1 follows in a second call.
            _, k0 = ctx.call(
                "rotate_new", q[:, :, :1], k[:, :, :1], positions_for(ctx, cache_lengths, 1), 1e4
            )
            q1, k1 = ctx.call(
                "rotate_new",
                q[:, :, 1:],
                k[:, :, 1:],
                positions_for(ctx, cache_lengths + 1, 1),
                1e4,
            )
            keys = np.concatenate((k0, k1), axis=2)
            return np.einsum("bhd,bhsd->bhs", q1[:, :, 0], keys)

        check.allclose(
            scores(base_len + shift),
            scores(base_len),
            rtol=1e-3,
            atol=1e-3,
            label="q.k scores after lengthening each row's cache",
        )

    check.forall(prop, trials=4)


@case("deep into a long context: cache lengths 1000 and 2040")
def long_context(ctx):
    start = lengths(1000, 2040)
    positions = positions_for(ctx, start, 4)
    ctx.check(
        "rotate_new", ctx.randn(2, 3, 4, 8), ctx.randn(2, 3, 4, 8), positions, 10000.0, **LOOSE
    )
    check_chunking(ctx, ctx.randn(2, 3, 4, 8), ctx.randn(2, 3, 4, 8), start, [2, 2], 10000.0, LOOSE)


@case("other sizes: one head with D = 2, and B = 1 with D = 16")
def other_sizes(ctx):
    positions = lengths(5, 0)[:, None] + np.arange(2)
    ctx.check(
        "rotate_new", ctx.randn(2, 1, 2, 2), ctx.randn(2, 1, 2, 2), positions, 10000.0, **TIGHT
    )
    ctx.check(
        "rotate_new",
        ctx.randn(1, 2, 2, 16),
        ctx.randn(1, 2, 2, 16),
        positions[:1],
        10000.0,
        **TIGHT,
    )
