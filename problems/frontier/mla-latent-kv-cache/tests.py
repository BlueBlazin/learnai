"""Tests for mla-latent-kv-cache.

The definition is the explicit formulation of Multi-Head Latent Attention: `explicit`
below computes it in numpy for a whole sequence at once, with full per-head keys and
values built from the up-projections, exactly as the statement writes it. The
solution's `mla_step` never sees the up-projections; it gets the matrices `absorb`
produces and a cache of latents. Cases compare its outputs and its cache, for
sequences fed in chunks, with that definition.

`absorb_np` is the test's own absorption, so that cases about `mla_step` do not
depend on the solution's `absorb`; the end-to-end cases use the solution's.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import ops

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints

# name: (d_model, n_heads, d_head, d_rope, d_c, d_cq)
GENERAL = (20, 4, 6, 4, 10, 14)
SQUARE = (12, 3, 4, 4, 12, 12)  # d_model = d_c = d_cq = n_heads * d_head: hides transpositions
ONE_HEAD = (8, 1, 8, 2, 4, 6)


def make_params(rng, config):
    d, heads, d_head, d_rope, d_c, d_cq = config

    def weight(out, inp):
        return (rng.standard_normal((out, inp)) * inp**-0.5).astype(np.float32)

    return {
        "w_dq": weight(d_cq, d),
        "w_uq": weight(heads * d_head, d_cq),
        "w_qr": weight(heads * d_rope, d_cq),
        "w_dkv": weight(d_c, d),
        "w_uk": weight(heads * d_head, d_c),
        "w_uv": weight(heads * d_head, d_c),
        "w_kr": weight(d_rope, d),
        "w_o": weight(d, heads * d_head),
        "n_heads": heads,
        "d_head": d_head,
    }


def absorb_np(p):
    heads = p["n_heads"]
    uq = p["w_uq"].reshape(heads, -1, p["w_uq"].shape[1])
    uk = p["w_uk"].reshape(heads, -1, p["w_uk"].shape[1])
    uv = p["w_uv"].reshape(heads, -1, p["w_uv"].shape[1])
    o = p["w_o"].reshape(p["w_o"].shape[0], heads, -1).transpose(1, 0, 2)
    return np.swapaxes(uk, 1, 2) @ uq, o @ uv


def step_weights(p, w_qk, w_ov):
    """What `mla_step` receives: no up-projection, no output projection."""
    keep = ("w_dq", "w_qr", "w_dkv", "w_kr", "n_heads", "d_head")
    return {**{name: p[name] for name in keep}, "w_qk": w_qk, "w_ov": w_ov}


def explicit(x, p):
    """MLA as defined: per-head keys and values, a whole sequence, causal. numpy."""
    batch, seq, _ = x.shape
    heads, d_head = p["n_heads"], p["d_head"]
    positions = np.arange(seq)

    def split(t):  # (B, T, H * n) -> (B, H, T, n)
        return t.reshape(batch, seq, heads, -1).transpose(0, 2, 1, 3)

    c_q = x @ p["w_dq"].T
    q_content = split(c_q @ p["w_uq"].T)
    q_rope = ops.rope(split(c_q @ p["w_qr"].T), positions)
    c = x @ p["w_dkv"].T
    k_content = split(c @ p["w_uk"].T)
    k_rope = ops.rope(x @ p["w_kr"].T, positions)  # (B, T, d_rope): one for all heads
    values = split(c @ p["w_uv"].T)
    q = np.concatenate((q_content, q_rope), axis=-1)
    k = np.concatenate((k_content, np.broadcast_to(k_rope[:, None], q_rope.shape)), axis=-1)
    # ops.attention divides by sqrt of the query width, which is d_head + d_rope here.
    o = ops.attention(q, k, values, np.tril(np.ones((seq, seq), dtype=bool)))
    out = o.transpose(0, 2, 1, 3).reshape(batch, seq, heads * d_head) @ p["w_o"].T
    return out.astype(np.float32), c, k_rope


def floats_in(obj):
    if isinstance(obj, np.ndarray):
        return obj.size
    if isinstance(obj, (tuple, list)):
        return sum(floats_in(item) for item in obj)
    if isinstance(obj, dict):
        return sum(floats_in(item) for item in obj.values())
    return 0


def feed(ctx, x, weights, sizes, cache=None, start=0):
    """Feed `x[:, start:]` chunk by chunk. Returns (outputs, final cache)."""
    batch = x.shape[0]
    d_c, d_rope = weights["w_dkv"].shape[0], weights["w_kr"].shape[0]
    outs = []
    for size in sizes:
        stop = start + size
        result = ctx.call("mla_step", x[:, start:stop], weights, cache)
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            "mla_step: must return a tuple (out, cache)",
            f"got {type(result).__name__}",
        )
        out, cache = result
        check.shape(out, x[:, start:stop].shape, label=f"out for tokens {start} to {stop - 1}")
        check.that(
            isinstance(cache, tuple)
            and len(cache) == 2
            and all(isinstance(part, np.ndarray) and part.ndim == 3 for part in cache),
            "mla_step: the returned cache must be a tuple (c, k_rope) of two 3-D tensors",
        )
        per_token = floats_in(cache) / (batch * stop)
        # The point of MLA: what is kept per token and layer is d_c + d_rope numbers.
        check.at_most("cache_floats_per_token", per_token, d_c + d_rope, unit="floats")
        check.shape(cache[0], (batch, stop, d_c), label=f"cached latents after {stop} tokens")
        check.shape(cache[1], (batch, stop, d_rope), label=f"cached RoPE keys after {stop} tokens")
        outs.append(out)
        start = stop
    return np.concatenate(outs, axis=1), cache


def check_sequence(ctx, x, p, weights, sizes):
    want_out, want_c, want_kr = explicit(x, p)
    out, (c, k_rope) = feed(ctx, x, weights, sizes)
    name = "+".join(str(s) for s in sizes)
    check.allclose(out, want_out, **TOL, label=f"outputs with chunks {name}")
    check.allclose(c, want_c, **TOL, label=f"cached latents after chunks {name}")
    check.allclose(k_rope, want_kr, **TOL, label=f"cached RoPE keys after chunks {name}")


def example_params():
    def m(rows):
        return np.array(rows, dtype=np.float32)

    eye = [[1.0, 0.0], [0.0, 1.0]]
    return {
        "w_dq": m(eye),
        "w_uq": m(eye),
        "w_qr": m([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "w_dkv": m(eye),
        "w_uk": m([[1.0, 1.0], [0.0, 2.0]]),
        "w_uv": m(eye),
        "w_kr": m([[1.0, 1.0], [0.0, 0.0]]),
        "w_o": m([[1.0, 1.0], [0.0, 1.0]]),
        "n_heads": 2,
        "d_head": 1,
    }


@case("the worked example: absorb, then two tokens fed one at a time", public=True)
def worked_example(ctx):
    p = example_params()
    w_qk, w_ov = ctx.check("absorb", p["w_uq"], p["w_uk"], p["w_uv"], p["w_o"], 2, **TOL)
    check.allclose(w_qk, [[[1, 0], [1, 0]], [[0, 0], [0, 2]]], atol=1e-6, label="w_qk")
    check.allclose(w_ov, [[[1, 0], [0, 0]], [[0, 1], [0, 1]]], atol=1e-6, label="w_ov")
    x = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    weights = step_weights(p, *absorb_np(p))
    out, cache = feed(ctx, x, weights, (1, 1))
    check.allclose(out, [[[1.0, 0.0], [1.305385, 0.805385]]], atol=1e-5, label="out")
    check.allclose(cache[0], [[[1.0, 0.0], [0.0, 1.0]]], atol=1e-6, label="cached latents")
    check.allclose(
        cache[1], [[[1.0, 0.0], [0.540302, 0.841471]]], atol=1e-5, label="cached RoPE keys"
    )
    check.allclose(out, explicit(x, p)[0], **TOL, label="out against the explicit formulation")


@case("absorb: random weights, 4 heads; scores and outputs keep their values", public=True)
def absorb_random(ctx):
    p = make_params(ctx.rng, GENERAL)
    heads, d_head = p["n_heads"], p["d_head"]
    w_qk, w_ov = ctx.check("absorb", p["w_uq"], p["w_uk"], p["w_uv"], p["w_o"], heads, **TOL)
    # The meaning, independent of the oracle. For any query latent and cached latent,
    # head h's content score is unchanged ...
    c_q, c = ctx.randn(p["w_uq"].shape[1]), ctx.randn(p["w_uk"].shape[1])
    q = (p["w_uq"] @ c_q).reshape(heads, d_head)
    k = (p["w_uk"] @ c).reshape(heads, d_head)
    check.allclose(w_qk @ c_q @ c, (q * k).sum(-1), **TOL, label="(w_qk[h] c_q) . c, per head")
    # ... and so is the output, for any latents the heads mixed.
    mixed = ctx.randn(heads, p["w_uv"].shape[1])
    v = np.stack([p["w_uv"].reshape(heads, d_head, -1)[h] @ mixed[h] for h in range(heads)])
    want = p["w_o"] @ v.reshape(-1)
    got = sum(w_ov[h] @ mixed[h] for h in range(heads))
    check.allclose(got, want, **TOL, label="sum over heads of w_ov[h] z_h")


@case("no cache: six tokens at once equal the explicit formulation", public=True)
def prefill_matches_explicit(ctx):
    p = make_params(ctx.rng, GENERAL)
    x = ctx.randn(2, 6, GENERAL[0])
    check_sequence(ctx, x, p, step_weights(p, *absorb_np(p)), (6,))


@case("four tokens, then two decode steps with the cache", public=True)
def decode_steps(ctx):
    # The new token is rotated for its absolute position, and attends to the cache.
    p = make_params(ctx.rng, GENERAL)
    x = ctx.randn(2, 6, GENERAL[0])
    check_sequence(ctx, x, p, step_weights(p, *absorb_np(p)), (4, 1, 1))


@case("the cache holds d_c + d_rope floats per token, whatever the number of heads", public=True)
def cache_size(ctx):
    # 10 + 4 = 14 floats per token, where per-head keys and values would take 2 * 4 * 6 = 48.
    # `feed` checks the size of the cache after every call.
    p = make_params(ctx.rng, GENERAL)
    x = ctx.randn(2, 6, GENERAL[0])
    check_sequence(ctx, x, p, step_weights(p, *absorb_np(p)), (4, 1, 1))
    _, cache = feed(ctx, x, step_weights(p, *absorb_np(p)), (6,))
    check.that(
        floats_in(cache) == 2 * 6 * (10 + 4),
        "mla_step: the cache does not hold d_c + d_rope floats per token",
        f"it holds {floats_in(cache)} floats for 2 sequences of 6 tokens",
    )


@case("end to end with the solution's own absorb, on a model whose matrices are all square")
def end_to_end_square(ctx):
    # d_model = d_c = d_cq = n_heads * d_head: a transposed or mis-split product has
    # the right shape here and the wrong values.
    p = make_params(ctx.rng, SQUARE)
    result = ctx.call("absorb", p["w_uq"], p["w_uk"], p["w_uv"], p["w_o"], p["n_heads"])
    check.that(isinstance(result, tuple) and len(result) == 2, "absorb: must return (w_qk, w_ov)")
    w_qk, w_ov = result
    check.shape(w_qk, (3, 12, 12), label="w_qk")
    check.shape(w_ov, (3, 12, 12), label="w_ov")
    x = ctx.randn(2, 6, SQUARE[0])
    check_sequence(ctx, x, p, step_weights(p, w_qk, w_ov), (6,))
    want_qk, want_ov = absorb_np(p)
    check.allclose(w_qk, want_qk, **TOL, label="w_qk")
    check.allclose(w_ov, want_ov, **TOL, label="w_ov")


@case("end to end on the 4-head model: absorb shapes, then chunks 2 + 4")
def end_to_end_general(ctx):
    p = make_params(ctx.rng, GENERAL)
    w_qk, w_ov = ctx.check("absorb", p["w_uq"], p["w_uk"], p["w_uv"], p["w_o"], p["n_heads"], **TOL)
    x = ctx.randn(2, 6, GENERAL[0], scale=2.0)
    check_sequence(ctx, x, p, step_weights(p, w_qk, w_ov), (2, 4))


@case("any way of cutting a sequence into chunks gives the explicit result")
def chunking_property(ctx):
    chunkings = [(6,), (2, 4), (4, 1, 1)]

    def prop(rng):
        p = make_params(rng, GENERAL)
        x = ctx.randn(2, 6, GENERAL[0], scale=float(rng.choice([0.5, 1.0, 3.0])))
        sizes = chunkings[int(rng.integers(len(chunkings)))]
        check_sequence(ctx, x, p, step_weights(p, *absorb_np(p)), sizes)

    check.forall(prop, trials=6)


@case("a 30-token sequence: 28 tokens, then two more far from position 0")
def far_positions(ctx):
    p = make_params(ctx.rng, GENERAL)
    x = ctx.randn(1, 30, GENERAL[0])
    weights = step_weights(p, *absorb_np(p))
    want_out, want_c, want_kr = explicit(x, p)
    out, (c, k_rope) = feed(ctx, x, weights, (28, 2))
    loose = {"rtol": 1e-3, "atol": 1e-4}
    check.allclose(out, want_out, **loose, label="outputs")
    check.allclose(c, want_c, **loose, label="cached latents")
    check.allclose(k_rope, want_kr, **loose, label="cached RoPE keys")


@case("one cache continued twice: the cache passed in is not changed")
def fork(ctx):
    p = make_params(ctx.rng, GENERAL)
    weights = step_weights(p, *absorb_np(p))
    x = ctx.randn(2, 4, GENERAL[0])
    alt = x.copy()
    alt[:, 3] = ctx.randn(2, GENERAL[0])
    _, cache = feed(ctx, x, weights, (3,))
    for seq, name in ((x, "first"), (alt, "second")):
        out, _ = feed(ctx, seq, weights, (1,), cache=cache, start=3)
        check.allclose(
            out, explicit(seq, p)[0][:, 3:], **TOL, label=f"out of the {name} continuation"
        )


@case("a single head, batch of one, and sharp attention (large inputs)")
def one_head_sharp(ctx):
    p = make_params(ctx.rng, ONE_HEAD)
    x = ctx.randn(1, 4, ONE_HEAD[0], scale=4.0)
    check_sequence(ctx, x, p, step_weights(p, *absorb_np(p)), (3, 1))
