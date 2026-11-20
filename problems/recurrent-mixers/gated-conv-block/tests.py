"""Tests for gated-conv-block.

Numeric cases compare with the oracle; three compare with arrays computed by
`Lfm3ShortConv` of `transformers` with the kernel length of LFM3 (`fixtures/shipped.npz`,
written by `fixtures/make.py`). Kernels are random, so they are not symmetric and a
flipped kernel gives different numbers.

Every stateful case feeds the solution its own returned state, and the chunkings
include chunks of 1, 2, and 3 tokens, shorter than the L - 1 = 6 rows of state.
Shapes are few (JAX compiles once per shape): width 6, kernel 7, 12 tokens.
"""

import numpy as np

from learnai.judge import case, check

FN = "conv_block"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
D, L, STEPS = 6, 7, 12


def weights(ctx, d=D, kernel=L):
    return (
        ctx.randn(3 * d, d, scale=d**-0.5),
        ctx.randn(d, kernel, scale=kernel**-0.5),
        ctx.randn(d, d, scale=d**-0.5),
    )


def compare(got, want, what=""):
    """`what` goes into the failure message: public cases pass one, hidden cases do not."""
    what = f"{what}: " if what else ""
    y, state = got
    check.shape(y, want[0].shape, label=f"{what}y")
    check.shape(state, want[1].shape, label=f"{what}state")
    check.allclose(y, want[0], **TOL, label=f"{what}y")
    check.allclose(state, want[1], **TOL, label=f"{what}state")


def run_chunks(ctx, x, w, sizes, state=None):
    """Feed `x` in consecutive chunks, handing each call the state of the one before."""
    assert sum(sizes) == len(x)
    ys, start = [], 0
    for n in sizes:
        y, state = ctx.call(FN, x[start : start + n], *w, state)
        check.shape(y, (n, x.shape[1]), label="y of one call")
        ys.append(y)
        start += n
    return np.concatenate(ys), state


@case(
    "the worked example: width 1, kernel (1, 10, 100), three tokens and then a fourth", public=True
)
def worked_example(ctx):
    w = (
        np.array([[1.0], [2.0], [1.0]], np.float32),
        np.array([[1.0, 10.0, 100.0]], np.float32),
        np.array([[1.0]], np.float32),
    )
    x = np.array([[1.0], [2.0], [3.0]], np.float32)
    y, state = ctx.check(FN, x, *w, None, **TOL)
    check.allclose(y.reshape(-1), np.array([200.0, 1640.0, 5646.0], np.float32), **TOL, label="y")
    check.allclose(state.reshape(-1), np.array([4.0, 9.0], np.float32), **TOL, label="state")
    y, state = ctx.check(FN, np.array([[1.0]], np.float32), *w, state, **TOL)
    check.allclose(
        y.reshape(-1), np.array([388.0], np.float32), **TOL, label="y of the fourth token"
    )
    check.allclose(state.reshape(-1), np.array([9.0, 1.0], np.float32), **TOL, label="state")


@case("12 tokens in one call: width 6, kernel 7", public=True)
def one_pass(ctx):
    ctx.check(FN, ctx.randn(STEPS, D), *weights(ctx), None, **TOL)


# Rejects kernel_flipped against the library itself.
@case("the block of LFM3 in Hugging Face Transformers: 12 tokens in one call", public=True)
def shipped_one_pass(ctx):
    f = ctx.fixture("shipped.npz")
    y, _state = ctx.call(FN, f["x"], f["w_in"], f["w_conv"], f["w_out"], None)
    check.allclose(y, f["y_one_pass"], **TOL, label="y")


# Rejects state_reset, state_stores_x, state_from_conv_output, and (through the chunks
# shorter than six tokens) state_from_new_rows_only.
@case("any chunking of 12 tokens gives the outputs and the state of one call", public=True)
def chunking(ctx):
    w, x = weights(ctx), ctx.randn(STEPS, D)
    want = ctx.expected(FN, x, *w, None)
    for sizes in ([5, 7], [1] * 12, [2, 3, 7], [3, 2, 2, 5]):
        compare(run_chunks(ctx, x, w, sizes), want, f"chunks {sizes}")


# Rejects symmetric_padding: with a centered window, output t reads tokens t + 1 .. t + 3.
@case(
    "causality: changing tokens 6 to 11 leaves the outputs of tokens 0 to 5 unchanged", public=True
)
def causal(ctx):
    def prop(rng):
        w, x = weights(ctx), ctx.randn(STEPS, D)
        other = x.copy()
        other[6:] = ctx.randn(6, D)
        y, _ = ctx.call(FN, x, *w, None)
        y_other, _ = ctx.call(FN, other, *w, None)
        check.allclose(y_other[:6], y[:6], rtol=0, atol=1e-6, label="outputs of tokens 0 to 5")
        check.that(
            not np.allclose(y_other[6:], y[6:], atol=1e-3),
            "outputs of tokens 6 to 11: unchanged although their inputs changed",
        )

    check.forall(prop, trials=3)


# Rejects state_in_place: the caller keeps the state it passed in.
@case("the tensors passed in, the state included, are not modified", public=True)
def inputs_unchanged(ctx):
    arrays = (ctx.randn(5, D), *weights(ctx), ctx.randn(L - 1, D))
    tensors = [ctx.fw.tensor(a) for a in arrays]
    ctx.solution.conv_block(*tensors)
    names = ("x", "w_in", "w_conv", "w_out", "state")
    for name, tensor, before in zip(names, tensors, arrays, strict=True):
        check.allclose(ctx.fw.numpy(tensor), before, rtol=0, atol=0, label=f"{name} after the call")


@case("the library with a cache: 5 tokens, then 7 calls of one token; outputs and final state")
def shipped_cached(ctx):
    f = ctx.fixture("shipped.npz")
    w = (f["w_in"], f["w_conv"], f["w_out"])
    compare(run_chunks(ctx, f["x"], w, [5] + [1] * 7), (f["y_cached"], f["state"]))


@case("the state after 1, 2, and 3 tokens of a new sequence: zeros, then rows of u")
def short_first_call(ctx):
    w, x = weights(ctx), ctx.randn(STEPS, D)
    b, _c, h = np.split(x @ w[0].T, 3, axis=-1)
    for n in (1, 2, 3):
        got = ctx.call(FN, x[:n], *w, None)
        compare(got, ctx.expected(FN, x[:n], *w, None))
        rows = np.concatenate((np.zeros((L - 1 - n, D), np.float32), (b * h)[:n]))
        check.allclose(got[1], rows, **TOL, label="state")


# Rejects the state mutants without relying on the solution's own state.
@case("continuing from a state the test supplies")
def supplied_state(ctx):
    w, state = weights(ctx), ctx.randn(L - 1, D)
    for n in (1, 5):
        x = ctx.randn(n, D)
        compare(ctx.call(FN, x, *w, state), ctx.expected(FN, x, *w, state))


# One row of state, and a kernel of even length (no center tap).
@case("other kernels: 2 taps and 4 taps")
def other_kernels(ctx):
    for kernel in (2, 4):
        w, x = weights(ctx, kernel=kernel), ctx.randn(STEPS, D)
        want = ctx.expected(FN, x, *w, None)
        compare(ctx.call(FN, x, *w, None), want)
        compare(run_chunks(ctx, x, w, [2, 3, 7]), want)


# A kernel that is 1 on the last tap only: the convolution is the identity and
# y = (C * B * h) W_out^T. Any other tap alignment shifts the sequence.
@case("a kernel with only its last tap set copies u to the current position")
def last_tap_is_now(ctx):
    w_in, _w_conv, w_out = weights(ctx)
    w_conv = np.zeros((D, L), np.float32)
    w_conv[:, -1] = 1.0
    x = ctx.randn(STEPS, D)
    y, _ = ctx.check(FN, x, w_in, w_conv, w_out, None, **TOL)
    b, c, h = np.split(x @ w_in.T, 3, axis=-1)
    check.allclose(y, (c * b * h) @ w_out.T, **TOL, label="y")
