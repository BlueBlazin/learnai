"""Tests for gated-delta-chunked.

`step_loop` below is the definition: the one-token rule of gated-delta-step applied T
times, in float64 numpy. Every case compares the solution with it; two cases also
compare with arrays computed by `torch_chunk_gated_delta_rule` of `transformers`
(`fixtures/shipped.npz`, written by `fixtures/make.py`).

The memory is handed over as a `Memory`: the kit's `DeltaMemory` behind the kit's
`Guarded`. It can be read and updated only through batched calls, each counted once
whatever its number of rows, and every case checks the counts against the number of
chunks: a token-by-token loop needs one read per token however it is written or
compiled. One case uses keys wider than all the rows the limits allow, so the memory
cannot be copied out and processed elsewhere.

Inputs are what the layer produces: unit-length keys, queries of length Dk^-0.5, beta
in [0, 1], log-decays at or below 0. Shapes are few because JAX compiles once per
shape, and chunks have at most 16 positions.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import DeltaMemory, Guarded, check_guard, guard

FN = "gdn_chunked"
TOL = {"rtol": 1e-3, "atol": 1e-4}  # the tolerance stated in Constraints
SMALL = (13, 3, 4, 5)  # T, H, Dk, Dv
NAMES = ("q", "k", "v", "log_g", "beta", "S0")


class Memory(Guarded, DeltaMemory):
    """The memory the solution gets: its counters, its array, and `peek` are refused."""

    label = "memory"
    refused = Guarded.refused | {"peek"}


def step_loop(q, k, v, log_g, beta, memory):
    """T steps of the gated delta rule: decay, write, read. float64."""
    q, k, v, log_g, beta, memory = (
        np.asarray(t, np.float64) for t in (q, k, v, log_g, beta, memory)
    )
    ys = []
    for t in range(len(q)):
        memory = memory * np.exp(log_g[t])[:, None, None]
        delta = beta[t][:, None] * (v[t] - np.einsum("hvk,hk->hv", memory, k[t]))
        memory = memory + delta[:, :, None] * k[t][:, None, :]
        ys.append(np.einsum("hvk,hk->hv", memory, q[t]))
    return np.stack(ys).astype(np.float32), memory.astype(np.float32)


def draw(ctx, steps, heads, d_k, d_v, *, slowest=0.4, zero_state=False):
    """Inputs as a delta layer produces them; per-token log-decay in [-slowest, 0]."""

    def unit(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    state = ctx.randn(heads, d_v, d_k, scale=0.5)
    return (
        (unit(ctx.randn(steps, heads, d_k)) * d_k**-0.5).astype(np.float32),
        unit(ctx.randn(steps, heads, d_k)).astype(np.float32),
        ctx.randn(steps, heads, d_v),
        -ctx.uniform(steps, heads, low=0.0, high=slowest),
        ctx.uniform(steps, heads, low=0.05, high=0.95),
        0 * state if zero_state else state,
    )


def call(ctx, args, chunk_size):
    """Call the solution on a fresh Memory; returns Y, the final state, the counters."""
    start = Memory(ctx.fw, args[5], max_rows=2 * chunk_size)
    guard(ctx)
    got_y, memory = ctx.call(FN, *args[:5], start, chunk_size)
    check_guard()
    check.that(
        isinstance(memory, Memory) and memory.counters is start.counters,
        "the second result is not a memory obtained from the given one by update calls",
    )
    check.allclose(start.peek(), args[5], rtol=0, atol=0, label="the memory passed in")
    return got_y, memory.peek(), start.counters


def run(ctx, args, chunk_size, what=""):
    """Compare both results with the step loop, then the calls with the chunk count.

    `what` goes into the failure message: public cases pass one, hidden cases do not.
    """
    prefix = f"{what}: " if what else ""
    got_y, got_s, counters = call(ctx, args, chunk_size)
    want_y, want_s = step_loop(*args)
    check.shape(got_y, want_y.shape, label=f"{prefix}Y")
    check.allclose(got_y, want_y, **TOL, label=f"{prefix}Y", dtype=True)
    check.allclose(got_s, want_s, **TOL, label=f"{prefix}S")
    chunks = -(-len(args[0]) // chunk_size)
    check.cost(counters, "reads", at_most=chunks, unit="calls")
    check.cost(counters, "writes", at_most=chunks, unit="calls")
    return got_y, got_s


@case("the worked example: three tokens, one head, chunk_size = 2", public=True)
def worked_example(ctx):
    q = np.array([[[1, 0]], [[0, 1]], [[0.6, 0.8]]], np.float32)
    k = np.array([[[1, 0]], [[0.6, 0.8]], [[0, 1]]], np.float32)
    v = np.array([[[1.0]], [[2.0]], [[-1.0]]], np.float32)
    log_g = np.log(np.array([[0.5], [0.5], [1.0]], np.float32))
    beta = np.array([[1.0], [0.5], [1.0]], np.float32)
    state = np.array([[[2.0, -2.0]]], np.float32)
    got_y, got_s = run(ctx, (q, k, v, log_g, beta, state), 2, "chunk_size 2")
    check.allclose(got_y.reshape(-1), np.array([1.0, 0.34, -0.122], np.float32), **TOL, label="Y")
    check.allclose(got_s.reshape(-1), np.array([1.13, -1.0], np.float32), **TOL, label="S")


@case("13 tokens, 3 heads, chunk_size = 4: three full chunks and one token left over", public=True)
def partial_last_chunk(ctx):
    run(ctx, draw(ctx, *SMALL), 4, "chunk_size 4")


# Rejects no_chunk_decay_on_state and keys_not_decayed_to_chunk_end (more than one
# chunk), and pad_with_decay_zero (chunk_size 4 and 16 pad; chunk_size 1 does not).
@case("the result does not depend on chunk_size: 1, 4, and 16 for 13 tokens", public=True)
def any_chunk_size(ctx):
    args = draw(ctx, *SMALL)
    for chunk_size in (1, 4, 16):
        run(ctx, args, chunk_size, f"chunk_size {chunk_size}")


@case("Hugging Face Transformers, 40 tokens, chunk_size = 16, nonzero initial memory", public=True)
def shipped_long(ctx):
    f = ctx.fixture("shipped.npz")
    args = tuple(f[f"long_{name}"] for name in NAMES)
    got_y, got_s = run(ctx, args, int(f["long_chunk_size"]), "chunk_size 16")
    check.allclose(got_y, f["long_Y"], **TOL, label="Y against the library")
    check.allclose(got_s, f["long_S"], **TOL, label="S against the library")


# Rejects per_token_loop, two_reads_per_chunk, and copies_memory_out. Keys have 64
# channels; six chunks of four tokens may read 6 calls of 8 rows, 48 numbers per value
# channel, so the memory cannot be recovered from what the limits allow.
@case("24 tokens, keys of width 64, chunk_size = 4: at most 6 reads and 6 writes", public=True)
def wide_keys(ctx):
    run(ctx, draw(ctx, 24, 2, 64, 5), 4, "chunk_size 4")


@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    args = draw(ctx, *SMALL)
    tensors = [ctx.fw.tensor(a) for a in args[:5]]
    guard(ctx)
    ctx.solution.gdn_chunked(*tensors, Memory(ctx.fw, args[5], max_rows=8), 4)
    check_guard()
    for name, tensor, before in zip(NAMES, tensors, args, strict=False):
        check.allclose(ctx.fw.numpy(tensor), before, rtol=0, atol=0, label=f"{name} after the call")


@case("Hugging Face Transformers, 13 tokens, chunk_size = 4")
def shipped_short(ctx):
    f = ctx.fixture("shipped.npz")
    args = tuple(f[f"short_{name}"] for name in NAMES)
    got_y, got_s = run(ctx, args, int(f["short_chunk_size"]))
    check.allclose(got_y, f["short_Y"], **TOL, label="Y")
    check.allclose(got_s, f["short_S"], **TOL, label="S")


# Rejects s0_missing_from_correction and q_reads_undecayed_start only through a
# nonzero memory; with a zero memory and one chunk both are exact.
@case("an empty memory and a single chunk, then the same tokens from a full memory")
def empty_then_full(ctx):
    args = draw(ctx, *SMALL, zero_state=True)
    run(ctx, args, 16)
    run(ctx, (*args[:5], ctx.randn(3, 5, 4)), 16)


# Rejects mask_after_exp: over a chunk of 16 the log-decays sum to about -240, so
# exp(c_i - c_j) overflows for i < j and must never be multiplied by anything.
@case("fast decay: log-decays down to -30 per token, chunk_size = 16")
def fast_decay(ctx):
    f = ctx.fixture("shipped.npz")
    args = [f[f"long_{name}"] for name in NAMES]
    args[3] = -ctx.uniform(40, 3, low=0.0, high=30.0)
    got_y, got_s = run(ctx, tuple(args), 16)
    check.finite(got_y, label="Y")
    check.finite(got_s, label="S")


# Rejects strict_lower_read and no_beta_on_keys at the ends of the gates.
@case("beta = 1 everywhere and no decay; then beta = 0 everywhere")
def gate_extremes(ctx):
    q, k, v, log_g, _beta, state = draw(ctx, *SMALL)
    ones = np.ones((13, 3), np.float32)
    run(ctx, (q, k, v, 0 * log_g, ones, state), 4)
    # With beta = 1, g = 1, and q = k / sqrt(Dk), a token reads its own value back.
    got_y, _ = run(ctx, (k * 0.5, k, v, 0 * log_g, ones, state), 4)
    check.allclose(got_y, 0.5 * v, **TOL, label="Y")
    run(ctx, (q, k, v, log_g, 0 * ones, state), 4)


# The memory is the whole state: a sequence cut anywhere, also inside a chunk, and
# continued from the returned memory gives the result of one call.
@case("13 tokens as 13, as 4 + 9, and as 9 + 4, with chunk_size = 4")
def continued(ctx):
    args = draw(ctx, *SMALL)
    want_y, want_s = step_loop(*args)
    for cut in (4, 9):
        head = (*(a[:cut] for a in args[:5]), args[5])
        y1, mid, _ = call(ctx, head, 4)
        y2, last, _ = call(ctx, (*(a[cut:] for a in args[:5]), mid), 4)
        check.allclose(np.concatenate((y1, y2)), want_y, **TOL, label="Y")
        check.allclose(last, want_s, **TOL, label="S")


@case("random inputs, slow and fast heads, chunk sizes 1, 4, and 16")
def random_sweep(ctx):
    def prop(rng):
        slowest = float(rng.choice([0.01, 0.4, 3.0]))
        args = draw(ctx, *SMALL, slowest=slowest)
        run(ctx, args, int(rng.choice([1, 4, 16])))

    check.forall(prop, trials=6)
