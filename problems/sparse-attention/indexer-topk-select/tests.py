"""Tests for indexer-topk-select.

`index_scores` is compared with the oracle and, in two cases, with the scores that
`GlmMoeDsaIndexer` of `transformers` hands to its top-k (`fixtures/shipped.npz`, written
by `fixtures/make.py`). `topk_mask` is exact: its scores are an input, so nothing
depends on rounding, and `select` below is its definition in numpy.

Shapes are few (JAX compiles once per shape): masks are (6, 10) or (4, 4), score inputs
(6, 4, 8) against (10, 8).
"""

import numpy as np

from learnai.judge import case, check

SCORES, MASK = "index_scores", "topk_mask"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
T, S, H, D = 6, 10, 4, 8


def select(scores, visible, k):
    """The definition: per row, the min(k, visible) best visible entries, ties by index."""
    masked = np.where(visible, scores, -np.inf)
    order = np.argsort(-masked, axis=-1, kind="stable")[:, :k]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask & visible


def causal(queries, keys):
    """Queries are the last `queries` of `keys` tokens; each sees itself and the past."""
    first = keys - queries
    return np.arange(keys)[None, :] <= (first + np.arange(queries))[:, None]


def check_mask(ctx, scores, visible, k, what=""):
    """Call topk_mask and compare with the definition. `what` is for public cases only."""
    scores = np.asarray(scores, dtype=np.float32)
    got = ctx.call(MASK, scores, visible, k)
    label = f"{MASK}, {what}" if what else MASK
    check.shape(got, scores.shape, label=label)
    check.that(
        got.dtype == np.bool_, f"{label}: got dtype {got.dtype}, which is not a boolean mask"
    )
    check.equal(got, select(scores, visible, k), label=label)
    return got


@case("the worked example: two queries, three keys, two heads of width 4", public=True)
def worked_example(ctx):
    q_idx = np.array([[[2, 0, 0, 0], [0, 2, 0, 0]], [[2, 2, 0, 0], [0, 0, 2, 0]]], dtype=np.float32)
    k_idx = np.array([[1, 1, 0, 0], [-1, 3, 0, 0], [2, 0, 1, 0]], dtype=np.float32)
    w = np.array([[1, 1], [2, -1]], dtype=np.float32)
    got = ctx.check(SCORES, q_idx, k_idx, w, **TOL)
    r = np.sqrt(0.5)
    expected = np.array([[2 * r, 3 * r, 2 * r], [4 * r, 4 * r, 3 * r]], dtype=np.float32)
    check.allclose(got, expected, **TOL, label=SCORES)
    visible = np.array([[True, True, False], [True, True, True]])
    got = check_mask(ctx, expected, visible, 1, "k = 1")
    check.equal(got, np.array([[False, True, False], [True, False, False]]), label="k = 1")
    got = check_mask(ctx, expected, visible, 3, "k = 3")
    check.equal(got, visible, label="k = 3")


@case(
    "random inputs: 6 queries, 10 keys, 4 heads of width 8; then top-3 of those scores", public=True
)
def random_inputs(ctx):
    q_idx, k_idx, w = ctx.randn(T, H, D), ctx.randn(S, D), ctx.randn(T, H)
    ctx.check(SCORES, q_idx, k_idx, w, **TOL)
    # The mask is checked on the oracle's scores, so it does not hinge on rounding.
    scores = ctx.expected(SCORES, q_idx, k_idx, w)
    check_mask(ctx, scores, causal(T, S), 3, "k = 3, causal")


# Rejects ties_to_highest_index, threshold_keeps_ties, and uses_topk: every row has
# equal scores on both sides of the cut. Small integers, so the ties are exact.
@case("equal scores at the cut: the lowest index is selected", public=True)
def ties_at_the_cut(ctx):
    scores = ctx.randint(0, 3, (T, S)).astype(np.float32)
    scores[0] = 0.0  # a query whose heads are all switched off by the ReLU
    scores[1] = [1, 1, 1, 2, 1, 1, 0, 1, 1, 1]
    visible = np.ones((T, S), dtype=bool)
    for k in (1, 3, 6):
        check_mask(ctx, scores, visible, k, f"k = {k}")
    check_mask(ctx, scores, causal(T, S), 3, "k = 3, causal")


@case(
    "GlmMoeDsaIndexer (Hugging Face Transformers), GLM-5.3 indexer sizes: 32 heads of 128",
    public=True,
)
def shipped_indexer(ctx):
    f = ctx.fixture("shipped.npz")
    visible = f["visible"]
    got = ctx.call(SCORES, f["q_idx"], f["k_idx"], f["w"])
    check.shape(got, visible.shape, label=SCORES)
    check.allclose(
        got[visible],
        f["scores"][visible],
        **TOL,
        label=f"{SCORES} against the library, visible entries",
    )
    # The library's top-k indices, scattered and intersected with the causal mask.
    got = check_mask(
        ctx, np.where(visible, f["scores"], 0.0), visible, int(f["k"]), "library scores"
    )
    check.equal(got, f["mask"], label=f"{MASK} against the library's selection")


@case("k at least the number of visible entries: the mask equals `visible`", public=True)
def everything_selected(ctx):
    def prop(rng):
        scores = ctx.randn(T, S)
        visible = causal(T, S) if rng.random() < 0.5 else rng.random((T, S)) < 0.6
        k = int(rng.choice([int(visible.sum(-1).max()), S, S + 7]))
        got = ctx.call(MASK, scores, visible, k)
        check.equal(got, visible, label=f"{MASK} with k = {k}")

    check.forall(prop, trials=6)


# Rejects masks_scores_in_place.
@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    arrays = {"q_idx": ctx.randn(T, H, D), "k_idx": ctx.randn(S, D), "w": ctx.randn(T, H)}
    tensors = {name: ctx.fw.tensor(a) for name, a in arrays.items()}
    ctx.solution.index_scores(*tensors.values())
    arrays["scores"], arrays["visible"] = ctx.randn(T, S), causal(T, S)
    tensors["scores"] = ctx.fw.tensor(arrays["scores"])
    tensors["visible"] = ctx.fw.tensor(arrays["visible"])
    ctx.solution.topk_mask(tensors["scores"], tensors["visible"], 3)
    for name, before in arrays.items():
        after = ctx.fw.numpy(tensors[name])
        if before.dtype == np.bool_:
            check.equal(after, before, label=f"{name} after the call")
        else:
            check.allclose(after, before, rtol=0, atol=0, label=f"{name} after the call")


# Rejects topk_before_mask: in every row the best scores belong to invisible entries,
# so a top-k taken first spends its slots on them and keeps fewer than k visible ones.
@case("invisible entries with the highest scores")
def invisible_entries_score_highest(ctx):
    scores = ctx.randn(T, S)
    visible = causal(T, S)
    scores = np.where(visible, scores, scores + 10.0)
    for k in (1, 3):
        check_mask(ctx, scores, visible, k)


# Rejects masks_with_zero: every visible score is negative, every invisible one is
# positive before masking, and 0 would still beat the visible ones.
@case("negative visible scores")
def negative_scores(ctx):
    visible = ctx.rng.random((T, S)) < 0.6
    visible[:, 0] = True
    scores = np.where(visible, -ctx.uniform(T, S, low=0.5, high=5.0), 1.0)
    for k in (2, 4):
        check_mask(ctx, scores, visible, k)


# Rejects no_intersection (rows with fewer than k visible entries, and a row with none).
@case("rows with few visible entries and a row with none")
def few_visible(ctx):
    scores = ctx.randn(T, S)
    visible = np.zeros((T, S), dtype=bool)
    visible[0, 7] = True
    visible[1, [2, 9]] = True
    visible[3, :5] = True
    visible[4] = True
    visible[5, [0, 4, 5]] = True  # row 2 has no visible entry
    for k in (1, 4, S, S + 3):
        check_mask(ctx, scores, visible, k)


@case("one query against ten keys, as in a decoding step")
def single_query(ctx):
    scores = ctx.randn(1, S)
    visible = np.ones((1, S), dtype=bool)
    for k in (1, 4, 25):
        check_mask(ctx, scores, visible, k)
    scores[0, [1, 4, 8]] = scores[0].max() + 1.0  # three equal best scores
    check_mask(ctx, scores, visible, 2)


# -0.0 and 0.0 are equal scores: a head weight below zero times a ReLU output of zero
# is -0.0, so both zeros occur in real index scores.
@case("zeros of both signs")
def signed_zeros(ctx):
    scores = np.zeros((T, S), dtype=np.float32)
    scores[:, 1::2] = -0.0
    scores[2, 6] = 1.0
    scores[3, [0, 5]] = -1.0
    visible = np.ones((T, S), dtype=bool)
    for k in (1, 2, 5):
        check_mask(ctx, scores, visible, k)
    check_mask(ctx, scores, causal(T, S), 3)


# Rejects index_offset_tiebreak: equal scores near 1000, where a small offset per index
# is below float32 resolution and breaks no tie, and distinct scores 1e-7 apart, which
# such an offset reorders. Distinct scores are never equal, however close.
@case("equal large scores, and distinct scores that are very close")
def large_ties_and_close_scores(ctx):
    visible = np.ones((T, S), dtype=bool)
    large = np.full((T, S), 1000.0, dtype=np.float32)
    large[0, [2, 7]] = 1001.0
    large[1, [0, 5]] = 999.0
    large[2] = -1000.0
    large[3, 9] = 1000.5
    for k in (1, 3, 6):
        check_mask(ctx, large, visible, k)
    check_mask(ctx, large, causal(T, S), 3)
    close = (1e-3 + 1e-7 * np.arange(S, dtype=np.float64)).astype(np.float32)
    if not (np.diff(close) > 0).all():
        raise RuntimeError("authoring error: the close scores are not distinct in float32")
    rows = np.stack([close, close[::-1], -close, close, close[::-1], -close[::-1]])
    for k in (1, 4):
        check_mask(ctx, rows, visible, k)
    check_mask(ctx, rows, causal(T, S), 2)


# T = S = H = D: every axis has the same length, so a swapped axis is not a shape error.
@case("all axes the same length")
def all_axes_equal(ctx):
    q_idx, k_idx, w = ctx.randn(4, 4, 4), ctx.randn(4, 4), ctx.randn(4, 4)
    ctx.check(SCORES, q_idx, k_idx, w, **TOL)
    scores = ctx.randn(4, 4)
    check_mask(ctx, scores, causal(4, 4), 2)
    check_mask(ctx, scores.T.copy(), causal(4, 4).T.copy(), 2)


# Rejects relu_after_sum, abs_instead_of_relu, relu_on_weights, and ignores_head_weights
# on constructed signs: all dot products negative (every score is 0), then all head
# weights negative (no score is above 0).
@case("heads that are all off, and head weights that are all negative")
def constructed_signs(ctx):
    k_idx = np.abs(ctx.randn(S, D)) + 0.1
    q_pos = np.abs(ctx.randn(T, H, D)) + 0.1
    w = ctx.randn(T, H)
    ctx.check(SCORES, -q_pos, k_idx, w, **TOL)
    ctx.check(SCORES, q_pos, k_idx, -np.abs(w) - 0.1, **TOL)


@case("the shipped scores again, with other selection sizes")
def shipped_other_k(ctx):
    f = ctx.fixture("shipped.npz")
    visible = f["visible"]
    scores = np.where(visible, f["scores"], 0.0)
    for k in (1, 2, 9, 2048):
        check_mask(ctx, scores, visible, k)


@case("random scores and masks")
def random_sweep(ctx):
    def prop(rng):
        scores = ctx.randn(T, S, scale=float(rng.choice([1e-3, 1.0, 100.0])))
        visible = causal(T, S) if rng.random() < 0.5 else rng.random((T, S)) < 0.7
        k = int(rng.integers(1, S + 3))
        got = check_mask(ctx, scores, visible, k)
        counts = np.minimum(k, visible.sum(-1))
        check.equal(got.sum(-1), counts, label="the number of selected entries per row")

    check.forall(prop, trials=12)
    q_idx, k_idx, w = ctx.randn(T, H, D, scale=3.0), ctx.randn(S, D, scale=3.0), ctx.randn(T, H)
    ctx.check(SCORES, q_idx, k_idx, w, **TOL)
