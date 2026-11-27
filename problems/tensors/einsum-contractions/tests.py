"""Tests for einsum-contractions.

Every case names the function it tests. All three functions are float32 sums of
products. Worked examples use whole numbers, which are exact; random inputs lie in
[-2, 2] and are compared with a tolerance that accepts the sum taken in any order
(einsum, matmul, or an explicit product and sum).
"""

import numpy as np

from learnai.judge import case, check

TOL = {"rtol": 1e-4, "atol": 1e-4}


def floats(values):
    return np.array(values, dtype=np.float32)


def rand(ctx, *shape):
    return ctx.uniform(*shape, low=-2.0, high=2.0)


def scores_inputs(ctx, batch, heads, queries, keys, width):
    return rand(ctx, batch, heads, queries, width), rand(ctx, batch, heads, keys, width)


def bilinear_inputs(ctx, batch, d, e):
    return rand(ctx, batch, d), rand(ctx, d, e), rand(ctx, batch, e)


def project_inputs(ctx, batch, heads, steps, d, e):
    return rand(ctx, batch, heads, steps, d), rand(ctx, heads, d, e)


# ------------------------------------------------------------------ attention_scores


# Rejects scores_transposed (shape (1, 1, 3, 2)) and scores_scaled (every value
# divided by sqrt(2)).
@case("attention_scores: the worked example, 2 queries and 3 keys of width 2", public=True)
def scores_example(ctx):
    q = floats([[[[1, 0], [0, 1]]]])
    k = floats([[[[1, 10], [2, 20], [3, 30]]]])
    got = ctx.check("attention_scores", q, k, **TOL)
    check.allclose(got, floats([[[[1, 2, 3], [10, 20, 30]]]]), label="attention_scores")


# Rejects scores_sum_over_key_heads: with three heads, every head would get the same
# scores.
@case("attention_scores: all sizes differ, B = 2, H = 3, T = 4, S = 5, D = 6", public=True)
def scores_distinct(ctx):
    ctx.check("attention_scores", *scores_inputs(ctx, 2, 3, 4, 5, 6), **TOL)


# scores_transposed returns the right shape here, and only the values differ.
@case("attention_scores: as many queries as keys, T = S = 4", public=True)
def scores_square(ctx):
    q = np.zeros((1, 1, 4, 2), dtype=np.float32)
    q[0, 0, :, 0] = [1, 2, 3, 4]
    k = np.zeros((1, 1, 4, 2), dtype=np.float32)
    k[0, 0, :, 0] = [1, 10, 100, 1000]
    got = ctx.check("attention_scores", q, k, **TOL)  # out[t, s] = (t + 1) * 10 ** s
    check.allclose(got[0, 0, 1], floats([2, 20, 200, 2000]), label="attention_scores[0, 0, 1]")
    ctx.check("attention_scores", *scores_inputs(ctx, 2, 3, 4, 4, 5), **TOL)


@case("attention_scores: all sizes equal, B = H = T = S = D = 3")
def scores_all_equal(ctx):
    for _ in range(2):
        ctx.check("attention_scores", *scores_inputs(ctx, 3, 3, 3, 3, 3), **TOL)


@case("attention_scores: the largest sizes, B = H = T = S = D = 8")
def scores_largest(ctx):
    ctx.check("attention_scores", *scores_inputs(ctx, 8, 8, 8, 8, 8), **TOL)


@case("attention_scores: a batch of one with one head, and a single query against 5 keys")
def scores_size_one(ctx):
    ctx.check("attention_scores", *scores_inputs(ctx, 1, 1, 4, 5, 6), **TOL)
    ctx.check("attention_scores", *scores_inputs(ctx, 2, 3, 1, 5, 6), **TOL)


@case("attention_scores: a single key, a width of 1, and all sizes 1")
def scores_size_one_more(ctx):
    ctx.check("attention_scores", *scores_inputs(ctx, 2, 3, 4, 1, 6), **TOL)
    ctx.check("attention_scores", *scores_inputs(ctx, 2, 3, 4, 5, 1), **TOL)
    ctx.check("attention_scores", *scores_inputs(ctx, 1, 1, 1, 1, 1), **TOL)


# -------------------------------------------------------------------------- bilinear


# Rejects bilinear_row_sums: row 0 of x against both rows of y gives 36 for 24.
@case("bilinear: the worked example, B = 2, D = 2, E = 3", public=True)
def bilinear_example(ctx):
    x = floats([[1, 2], [0, 1]])
    w = floats([[1, 2, 3], [4, 5, 6]])
    y = floats([[1, 0, 1], [0, 1, 0]])
    got = ctx.check("bilinear", x, w, y, **TOL)
    check.allclose(got, floats([24, 5]), label="bilinear")


# Rejects bilinear_bare_squeeze: the result must be (1,), and squeeze() returns a
# tensor with no axes.
@case("bilinear: a batch of one returns shape (1,)", public=True)
def bilinear_batch_of_one(ctx):
    x = floats([[1, 2, 3]])
    w = floats([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 100, 1000]])
    y = floats([[1, 1, 1, 2]])
    got = ctx.check("bilinear", x, w, y, **TOL)
    check.allclose(got, floats([6321]), label="bilinear")


# bilinear_w_transposed runs here, and returns x[b] @ w.T @ y[b].
@case("bilinear: a square w that is not symmetric, D = E = 4", public=True)
def bilinear_square_w(ctx):
    x = floats([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
    w = floats([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    y = floats([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    got = ctx.check("bilinear", x, w, y, **TOL)  # one-hot rows: out[b] is one entry of w
    check.allclose(got, floats([3, 20, 12]), label="bilinear")
    ctx.check("bilinear", *bilinear_inputs(ctx, 3, 4, 4), **TOL)


@case("bilinear: all sizes differ, B = 5, D = 3, E = 4, and all equal, B = D = E = 3")
def bilinear_random(ctx):
    for _ in range(2):
        ctx.check("bilinear", *bilinear_inputs(ctx, 5, 3, 4), **TOL)
        ctx.check("bilinear", *bilinear_inputs(ctx, 3, 3, 3), **TOL)


@case("bilinear: the largest sizes, B = D = E = 8")
def bilinear_largest(ctx):
    for _ in range(3):
        ctx.check("bilinear", *bilinear_inputs(ctx, 8, 8, 8), **TOL)


@case("bilinear: D = 1, E = 1, and all sizes 1")
def bilinear_size_one(ctx):
    ctx.check("bilinear", *bilinear_inputs(ctx, 5, 1, 4), **TOL)
    ctx.check("bilinear", *bilinear_inputs(ctx, 5, 3, 1), **TOL)
    ctx.check("bilinear", *bilinear_inputs(ctx, 1, 1, 1), **TOL)


# ------------------------------------------------------------------ per_head_project


# Rejects project_first_head_for_all: head 1 would return [4, 5, 6].
@case("per_head_project: the worked example, 2 heads with different matrices", public=True)
def project_example(ctx):
    x = floats([[[[1, 0]], [[0, 1]]]])
    w = floats([[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]])
    got = ctx.check("per_head_project", x, w, **TOL)
    check.allclose(got, floats([[[[1, 2, 3]], [[40, 50, 60]]]]), label="per_head_project")


@case("per_head_project: all sizes differ, B = 2, H = 3, T = 4, D = 5, E = 6", public=True)
def project_distinct(ctx):
    ctx.check("per_head_project", *project_inputs(ctx, 2, 3, 4, 5, 6), **TOL)


# project_w_transposed runs here, and multiplies by the transpose of each w[h].
@case("per_head_project: square matrices that are not symmetric, D = E = 5", public=True)
def project_square_w(ctx):
    ctx.check("per_head_project", *project_inputs(ctx, 2, 3, 4, 5, 5), **TOL)


# project_weight_by_position runs here, and uses w[t] for position t.
@case("per_head_project: as many heads as positions, H = T = 3, and all sizes equal")
def project_heads_equal_steps(ctx):
    ctx.check("per_head_project", *project_inputs(ctx, 2, 3, 3, 5, 6), **TOL)
    ctx.check("per_head_project", *project_inputs(ctx, 3, 3, 3, 3, 3), **TOL)


@case("per_head_project: the largest sizes, B = H = T = D = E = 8")
def project_largest(ctx):
    ctx.check("per_head_project", *project_inputs(ctx, 8, 8, 8, 8, 8), **TOL)


@case("per_head_project: a batch of one, a single head, and a single position")
def project_size_one(ctx):
    ctx.check("per_head_project", *project_inputs(ctx, 1, 3, 4, 5, 6), **TOL)
    ctx.check("per_head_project", *project_inputs(ctx, 2, 1, 4, 5, 6), **TOL)
    ctx.check("per_head_project", *project_inputs(ctx, 2, 3, 1, 5, 6), **TOL)


@case("per_head_project: D = 1, E = 1, and all sizes 1")
def project_size_one_more(ctx):
    ctx.check("per_head_project", *project_inputs(ctx, 2, 3, 4, 1, 6), **TOL)
    ctx.check("per_head_project", *project_inputs(ctx, 2, 3, 4, 5, 1), **TOL)
    ctx.check("per_head_project", *project_inputs(ctx, 1, 1, 1, 1, 1), **TOL)
