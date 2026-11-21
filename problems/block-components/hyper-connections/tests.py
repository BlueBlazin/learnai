"""Tests for hyper-connections.

Numeric cases compare with the oracle. Five cases compare with arrays computed by
`DeepseekV4HyperConnection` and `DeepseekV4DecoderLayer` of `transformers`
(`fixtures/shipped.npz`, written by `fixtures/make.py`).

The order of the Sinkhorn normalizations (columns first, columns last) and the axis of
the softmax stop mattering once the iteration has converged, so the cases that pin them
down use 1 and 2 iterations; 20 is the shipped count.

Shapes are few (JAX compiles once per shape): 5 tokens, 4 streams of width 8, and one
case each with 3 streams and with 4 streams of width 4.
"""

import numpy as np

from learnai.judge import case, check

WEIGHTS, MERGE = "hc_weights", "hc_merge"
TOL = {"rtol": 1e-4, "atol": 1e-6}  # the tolerance stated in Constraints
T, N, D = 5, 4, 8
EPS = 1e-6
PARTS = ("collapsed", "post", "comb")


def draw(ctx, n=N, dim=D, steps=T, stream_scale=1.0, gain=2.0):
    """(streams, fn, base, scale). Raw weights of order `gain`, base of order 1."""
    k = 2 * n + n * n
    return (
        ctx.randn(steps, n, dim, scale=stream_scale),
        ctx.randn(k, n * dim, scale=gain * (n * dim) ** -0.5),
        ctx.randn(k),
        ctx.uniform(3, low=0.5, high=1.5),
    )


def weights(ctx, args, iters, eps=EPS, **tol):
    """Call hc_weights and compare all three results with the oracle."""
    got = ctx.call(WEIGHTS, *args, iters, eps)
    check.that(
        isinstance(got, tuple) and len(got) == 3,
        f"{WEIGHTS}: the result is not a tuple (collapsed, post, comb)",
    )
    want = ctx.expected(WEIGHTS, *args, iters, eps)
    for name, g, w in zip(PARTS, got, want, strict=True):
        check.allclose(g, w, **(tol or TOL), label=name, dtype=True)
    return got


def shipped(ctx, prefix, iters, eps=EPS):
    f = ctx.fixture("shipped.npz")
    args = tuple(f[f"{prefix}_{name}"] for name in ("streams", "fn", "base", "scale"))
    got = ctx.call(WEIGHTS, *args, iters, eps)
    check.that(
        isinstance(got, tuple) and len(got) == 3,
        f"{WEIGHTS}: the result is not a tuple (collapsed, post, comb)",
    )
    for name, g in zip(PARTS, got, strict=True):
        check.allclose(g, f[f"{prefix}_{name}"], **TOL, label=name, dtype=True)


@case("the worked example: two streams, weights set by the base alone", public=True)
def worked_example(ctx):
    streams = np.array([[[1.0, 2.0], [3.0, -1.0]]], dtype=np.float32)
    fn = np.zeros((8, 4), dtype=np.float32)
    ln3 = np.log(3.0)
    base = np.array([0.0, ln3, 0.0, -ln3, 0.0, ln3, 0.0, 0.0], dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    collapsed, post, comb = weights(ctx, (streams, fn, base, scale), 1)
    check.allclose(collapsed, [[2.75, 0.25]], atol=1e-5, label="collapsed")
    check.allclose(post, [[1.0, 0.5]], atol=1e-5, label="post")
    check.allclose(comb, [[[1 / 3, 0.6], [2 / 3, 0.4]]], atol=1e-5, label="comb")
    _, _, comb2 = weights(ctx, (streams, fn, base, scale), 2)
    want = [[[0.363636, 0.631579], [0.636364, 0.368421]]]
    check.allclose(comb2, want, atol=1e-5, label="comb after 2 iterations")

    y = np.array([[10.0, 20.0]], dtype=np.float32)
    exact = np.float32
    merged = ctx.check(MERGE, streams, y, post.astype(exact), comb.astype(exact), **TOL)
    want = [[[12.333333, 20.0], [6.8, 10.8]]]
    check.allclose(merged, want, atol=1e-4, label=MERGE)


@case("hc_weights against DeepseekV4HyperConnection: 1, 2, and 20 iterations", public=True)
def shipped_iterations(ctx):
    for iters in (1, 2, 20):
        shipped(ctx, f"w{iters}", iters)


# With eps = 0.01 each of its uses changes the result by about a percent: in `collapsed`
# through pre, in `comb` through the softmax and the divisors, and not at all in
# `post`. The norm's own epsilon stays 1e-6, which the streams of order 1e-3 show.
@case("eps = 0.01 against DeepseekV4HyperConnection: streams of order 1 and 1e-3", public=True)
def shipped_large_eps(ctx):
    shipped(ctx, "eps", 2, eps=0.01)
    shipped(ctx, "eps_small", 2, eps=0.01)


@case("a decoder layer: both hyper-connections against DeepseekV4DecoderLayer", public=True)
def shipped_layer(ctx):
    f = ctx.fixture("shipped.npz")
    streams = f["layer_streams"]
    for name, y in (("attn", f["layer_y_attn"]), ("ffn", f["layer_y_mlp"])):
        args = tuple(f[f"layer_{name}_{part}"] for part in ("fn", "base", "scale"))
        result = ctx.call(WEIGHTS, streams, *args, 20, EPS)
        check.that(
            isinstance(result, tuple) and len(result) == 3,
            f"{WEIGHTS}: the result is not a tuple (collapsed, post, comb)",
        )
        _, post, comb = result
        streams = ctx.call(MERGE, streams, y, post, comb)
        check.shape(streams, (T, N, D), label=MERGE)
    check.allclose(
        streams, f["layer_out"], rtol=1e-4, atol=1e-5, label="the layer's output streams"
    )


def mild(args):
    """Rescale the comb part of `fn` and `base` so every token's N * N softmax inputs span
    at most 2 (largest minus smallest), where 20 Sinkhorn iterations have converged."""
    streams, fn, base, scale = (np.asarray(a, np.float64) for a in args)
    steps, n, _ = streams.shape
    flat = streams.reshape(steps, -1)
    flat = flat / np.sqrt((flat**2).mean(axis=-1, keepdims=True) + 1e-6)
    logits = (flat @ fn.T)[:, 2 * n :] * scale[2] + base[2 * n :]
    span = float((logits.max(axis=-1) - logits.min(axis=-1)).max())
    shrink = np.float32(min(1.0, 1.9 / span))
    fn, base = args[1].copy(), args[2].copy()
    fn[2 * n :] *= shrink
    base[2 * n :] *= shrink
    return args[0], fn, base, args[3]


@case("comb is doubly stochastic: columns always, rows once converged", public=True)
def doubly_stochastic(ctx):
    def prop(rng):
        args = mild(draw(ctx, gain=float(rng.choice([0.5, 1.0, 2.0]))))
        ones = np.ones((T, N), dtype=np.float32)
        for iters in (1, 2, 20):
            result = ctx.call(WEIGHTS, *args, iters, EPS)
            check.that(
                isinstance(result, tuple) and len(result) == 3,
                f"{WEIGHTS}: the result is not a tuple (collapsed, post, comb)",
            )
            comb = result[2]
            check.shape(comb, (T, N, N), label="comb")
            check.that(bool((comb > 0).all()), "comb: some entries are not positive")
            check.allclose(
                comb.sum(axis=-2),
                ones,
                rtol=0,
                atol=1e-5,
                label=f"column sums of comb after {iters} iterations",
            )
        check.allclose(
            comb.sum(axis=-1), ones, rtol=0, atol=1e-4, label="row sums of comb after 20 iterations"
        )

    check.forall(prop, trials=4)


@case("hc_merge with a comb that is not symmetric", public=True)
def merge_not_symmetric(ctx):
    streams, y = ctx.randn(T, N, D), ctx.randn(T, D)
    post = ctx.uniform(T, N, low=0.0, high=2.0)
    comb = ctx.uniform(T, N, N)
    ctx.check(MERGE, streams, y, post, comb, **TOL)


@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    arrays = dict(zip(("streams", "fn", "base", "scale"), draw(ctx), strict=True))
    tensors = {name: ctx.fw.tensor(a) for name, a in arrays.items()}
    _, post, comb = ctx.solution.hc_weights(*tensors.values(), 3, EPS)
    merge = {"y": ctx.randn(T, D), "post": ctx.fw.numpy(post), "comb": ctx.fw.numpy(comb)}
    arrays.update(merge)
    tensors.update({name: ctx.fw.tensor(a) for name, a in merge.items()})
    ctx.solution.hc_merge(tensors["streams"], tensors["y"], tensors["post"], tensors["comb"])
    for name, before in arrays.items():
        check.allclose(
            ctx.fw.numpy(tensors[name]), before, rtol=0, atol=0, label=f"{name} after the calls"
        )


@case("random streams and parameters at 1, 2, 3, and 20 iterations")
def random_iterations(ctx):
    for iters in (1, 2, 3, 20):
        args = draw(ctx)
        _, post, comb = weights(ctx, args, iters)
        ctx.check(MERGE, args[0], ctx.randn(T, D), post, comb, **TOL)


# Rejects norm_per_stream: one stream much larger than the others. Normalized as a
# whole, the small streams stay small in the flattened vector.
@case("streams of very different sizes")
def unequal_streams(ctx):
    for iters in (1, 20):
        streams, fn, base, scale = draw(ctx)
        streams = streams * np.array([10.0, 1.0, 0.1, 1.0], dtype=np.float32)[None, :, None]
        weights(ctx, (streams, fn, base, scale), iters)


# The public eps case again on other inputs, at 1 and 3 iterations.
@case("a larger eps")
def large_eps(ctx):
    weights(ctx, draw(ctx), 1, eps=0.01)
    weights(ctx, draw(ctx, stream_scale=1e-3), 3, eps=0.01)


@case("three streams")
def three_streams(ctx):
    shipped(ctx, "n3", 20)
    for iters in (1, 2):
        args = draw(ctx, n=3)
        _, post, comb = weights(ctx, args, iters)
        ctx.check(MERGE, args[0], ctx.randn(T, D), post, comb, **TOL)


# N = D = 4 and T = 4: every axis that could be confused has the same length.
@case("square: 4 tokens, 4 streams of width 4")
def square_shapes(ctx):
    for iters in (1, 2):
        args = draw(ctx, n=4, dim=4, steps=4)
        _, post, comb = weights(ctx, args, iters)
        ctx.check(MERGE, args[0], ctx.randn(4, 4), post, comb, **TOL)


# Sharp mixing matrices: raw comb values of order 6, so the softmax is close to one-hot
# and the entries that eps keeps alive are the small ones.
@case("large raw weights")
def sharp_weights(ctx):
    for iters in (1, 2, 20):
        weights(ctx, draw(ctx, gain=6.0), iters)


@case("one token; equal streams, as right after the embedding")
def copied_embedding(ctx):
    streams, fn, base, scale = draw(ctx, steps=1)
    streams = np.repeat(streams[:, :1], N, axis=1)
    _, post, comb = weights(ctx, (streams, fn, base, scale), 20)
    ctx.check(MERGE, streams, ctx.randn(1, D), post, comb, **TOL)
