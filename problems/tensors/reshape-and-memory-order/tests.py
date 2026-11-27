"""Tests for reshape-and-memory-order.

Every case names the function it tests. Numeric cases compare with the oracle. Property
cases state the definition without it, and that no function creates, drops, or repeats
a value: the output is a rearrangement of the input. Nothing is computed, so every
comparison is exact.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def counting(*shape, start=0):
    """start, start + 1, ... in the given shape: every element is recognizable."""
    return (start + np.arange(np.prod(shape), dtype=np.float32)).reshape(shape)


def distinct(rng, *shape, start=0):
    """Distinct whole numbers in a random order."""
    size = int(np.prod(shape))
    return (start + rng.permutation(size)).astype(np.float32).reshape(shape)


def check_rearranges(got, inputs, label):
    """`got` holds exactly the values of `inputs`, each once."""
    want = np.sort(np.concatenate([np.ravel(x) for x in inputs]))
    have = np.sort(np.ravel(got))
    check.that(
        have.shape == want.shape and np.array_equal(have, want),
        f"{label}: the output is not a rearrangement of the input values "
        f"({have.size} values out, {want.size} in)",
        f"sorted input values\n  {want.tolist()}\n\nsorted output values\n  {have.tolist()}",
    )


# --------------------------------------------------------------------- group_columns


# Rejects groups_along_rows and strided_groups.
@case("group_columns: the worked example, a (2, 6) matrix in groups of 3", public=True)
def group_example(ctx):
    got = ctx.check("group_columns", counting(2, 6), 3, **EXACT)
    expected = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=np.float32)
    check.allclose(got, expected, label="group_columns", **EXACT)


@case("group_columns: the same (2, 6) matrix in groups of 2", public=True)
def group_pairs(ctx):
    ctx.check("group_columns", counting(2, 6), 2, **EXACT)


# Every axis of the result has size 3: only the values tell the layouts apart.
@case("group_columns: R = C / g = g = 3, and a square (4, 4) matrix in groups of 2")
def group_equal_sizes(ctx):
    ctx.check("group_columns", counting(3, 9), 3, **EXACT)
    ctx.check("group_columns", counting(4, 4), 2, **EXACT)
    ctx.check("group_columns", ctx.randn(4, 4), 2, **EXACT)


@case("group_columns: one row, groups of 1, one group, and a (1, 1) matrix")
def group_size_one(ctx):
    ctx.check("group_columns", counting(1, 6), 2, **EXACT)
    ctx.check("group_columns", counting(2, 6), 1, **EXACT)
    ctx.check("group_columns", counting(2, 6), 6, **EXACT)
    ctx.check("group_columns", counting(1, 1), 1, **EXACT)


# ------------------------------------------------------------------------ interleave


# Rejects interleave_by_concat and interleave_stacks_in_front.
@case("interleave: the worked example, two vectors of 3", public=True)
def interleave_example(ctx):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    got = ctx.check("interleave", a, b, **EXACT)
    expected = np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0], dtype=np.float32)
    check.allclose(got, expected, label="interleave", **EXACT)


@case("interleave: leading axes are kept, inputs of shape (2, 3)", public=True)
def interleave_matrix(ctx):
    ctx.check("interleave", counting(2, 3), counting(2, 3, start=100), **EXACT)


@case("interleave: three axes, and all axes the same size")
def interleave_ranks(ctx):
    ctx.check("interleave", counting(2, 3, 4), counting(2, 3, 4, start=100), **EXACT)
    ctx.check("interleave", counting(3, 3), counting(3, 3, start=100), **EXACT)
    ctx.check("interleave", counting(2, 2, 2), counting(2, 2, 2, start=100), **EXACT)


@case("interleave: a last axis of 1, a leading axis of 1, and two single values")
def interleave_size_one(ctx):
    ctx.check("interleave", counting(4, 1), counting(4, 1, start=100), **EXACT)
    ctx.check("interleave", counting(1, 4), counting(1, 4, start=100), **EXACT)
    ctx.check("interleave", counting(1), counting(1, start=100), **EXACT)


# -------------------------------------------------------------------------- patchify


# Rejects patchify_reshape_only (which returns x itself here), patches_column_major,
# and patch_pixels_column_major.
@case("patchify: the worked example, one 4 by 4 image in 2 by 2 patches", public=True)
def patchify_example(ctx):
    got = ctx.check("patchify", counting(1, 4, 4), 2, **EXACT)
    expected = np.array(
        [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]], dtype=np.float32
    )
    check.allclose(got, expected, label="patchify", **EXACT)


@case("patchify: all sizes differ, x of shape (2, 4, 6) and p = 2", public=True)
def patchify_distinct(ctx):
    ctx.check("patchify", counting(2, 4, 6), 2, **EXACT)
    ctx.check("patchify", ctx.randn(2, 4, 6), 2, **EXACT)


@case("patchify: patches of 3 by 3, x of shape (1, 6, 9)")
def patchify_three(ctx):
    ctx.check("patchify", counting(1, 6, 9), 3, **EXACT)


# The result (4, 4, 4) has the shape of x: only the values tell the layouts apart.
@case("patchify: every axis of x and of the result has size 4")
def patchify_equal_sizes(ctx):
    ctx.check("patchify", counting(4, 4, 4), 2, **EXACT)
    ctx.check("patchify", ctx.randn(4, 4, 4), 2, **EXACT)


@case("patchify: one row of patches, one column of patches, and one patch")
def patchify_single_rows(ctx):
    ctx.check("patchify", counting(2, 2, 6), 2, **EXACT)
    ctx.check("patchify", counting(2, 6, 2), 2, **EXACT)
    ctx.check("patchify", counting(2, 3, 3), 3, **EXACT)


@case("patchify: patches of one pixel, a batch of one, and a (1, 1, 1) input")
def patchify_size_one(ctx):
    ctx.check("patchify", counting(2, 2, 6), 1, **EXACT)
    ctx.check("patchify", counting(1, 2, 6), 2, **EXACT)
    ctx.check("patchify", counting(1, 1, 1), 1, **EXACT)


# ------------------------------------------------------------------------ properties

GROUP_SHAPES = [(2, 6, 3), (2, 6, 2), (3, 9, 3), (4, 4, 2)]  # (R, C, g)
INTERLEAVE_SHAPES = [(3,), (2, 3), (2, 3, 4), (3, 3)]
PATCH_SHAPES = [(1, 4, 4, 2), (2, 4, 6, 2), (1, 6, 9, 3), (4, 4, 4, 2)]  # (B, H, W, p)


@case("group_columns: group k of row r is the slice w[r, k * g : (k + 1) * g]", public=True)
def group_definition(ctx):
    def prop(rng):
        rows, cols, g = GROUP_SHAPES[int(rng.integers(len(GROUP_SHAPES)))]
        w = distinct(rng, rows, cols)
        got = ctx.call("group_columns", w, g)
        check.shape(got, (rows, cols // g, g), label="group_columns")
        check_rearranges(got, [w], "group_columns")
        for k in range(cols // g):
            check.allclose(
                got[:, k],
                w[:, k * g : (k + 1) * g],
                label=f"group {k} of group_columns(w, {g}) for w of shape {w.shape}",
                **EXACT,
            )

    check.forall(prop, trials=6)


@case("interleave: even positions hold a, odd positions hold b", public=True)
def interleave_definition(ctx):
    def prop(rng):
        shape = INTERLEAVE_SHAPES[int(rng.integers(len(INTERLEAVE_SHAPES)))]
        both = distinct(rng, 2, *shape)
        a, b = both[0], both[1]
        got = ctx.call("interleave", a, b)
        check.shape(got, (*shape[:-1], 2 * shape[-1]), label="interleave")
        check_rearranges(got, [a, b], "interleave")
        where = f"for inputs of shape {shape}"
        check.allclose(got[..., 0::2], a, label=f"out[..., 0::2] of interleave {where}", **EXACT)
        check.allclose(got[..., 1::2], b, label=f"out[..., 1::2] of interleave {where}", **EXACT)

    check.forall(prop, trials=6)


@case("patchify: patch (i, j) is the block x[b, i*p : (i+1)*p, j*p : (j+1)*p], flattened")
def patchify_definition(ctx):
    def prop(rng):
        batch, height, width, p = PATCH_SHAPES[int(rng.integers(len(PATCH_SHAPES)))]
        x = distinct(rng, batch, height, width)
        got = ctx.call("patchify", x, p)
        rows, cols = height // p, width // p
        check.shape(got, (batch, rows * cols, p * p), label="patchify")
        check_rearranges(got, [x], "patchify")
        for i in range(rows):
            for j in range(cols):
                block = x[:, i * p : (i + 1) * p, j * p : (j + 1) * p]
                check.allclose(
                    got[:, i * cols + j],
                    block.reshape(batch, p * p),
                    label=f"patch {i * cols + j} (row {i}, column {j}) of patchify(x, {p}) "
                    f"for x of shape {x.shape}",
                    **EXACT,
                )

    check.forall(prop, trials=6)
