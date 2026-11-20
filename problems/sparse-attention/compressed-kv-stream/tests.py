"""Tests for compressed-kv-stream.

`window_entries` is the definition: the entries of a whole stream in float64 numpy.
Every case feeds a stream to the solution in pieces (`stream`) and checks, after each
call, the entries the call emitted and all three parts of the returned state. Three
cases also compare with what `DeepseekV4HCACompressor` and `DeepseekV4HCACache` of
`transformers` computed for the same stream (`fixtures/shipped.npz`, written by
`fixtures/make.py`).

Shapes are few (JAX compiles once per shape): synthetic streams have 14 rows of width
16 and a window of 4, cut one of five ways.
"""

import numpy as np

from learnai.judge import case, check

FN, VIS = "compress", "visible_entries"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
N, D, RATE = 14, 16, 4
WHOLE, PIECES, ROWS = [14], [3, 4, 13, 14], list(range(1, 15))
INSIDE, TWO = [2, 11, 14], [5, 14]


def window_entries(kv, gate, position_bias, gain, rate):
    """One entry per complete window of the whole stream. float64."""
    kv, gate, bias, gain = (np.asarray(a, np.float64) for a in (kv, gate, position_bias, gain))
    windows = kv.shape[0] // rate
    win_kv = kv[: windows * rate].reshape(windows, rate, -1)
    logits = gate[: windows * rate].reshape(windows, rate, -1) + bias
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    pooled = (weights * win_kv).sum(axis=1)
    normed = pooled / np.sqrt((pooled**2).mean(axis=-1, keepdims=True) + 1e-6)
    return (gain * normed).astype(np.float32)


def draw(ctx, rows=N, dim=D, rate=RATE, gate_scale=1.0):
    """A stream and parameters: (kv, gate, position_bias, gain)."""
    return (
        ctx.randn(rows, dim),
        ctx.randn(rows, dim, scale=gate_scale),
        ctx.randn(rate, dim),
        ctx.uniform(dim, low=0.5, high=1.5),
    )


def empty_state(dim):
    return (np.zeros((0, dim), np.float32), np.zeros((0, dim), np.float32), 0)


def stream(ctx, kv, gate, position_bias, gain, rate, cuts, public=False, state=None):
    """Feed rows [0, cuts[0]), [cuts[0], cuts[1]), ...; check every call.

    Returns all entries and the final state. Failure messages name the call only in
    public cases. `state` (with its rows prepended to the stream) starts mid-stream.
    """
    want = window_entries(kv, gate, position_bias, gain, rate)
    dim = kv.shape[1]
    state = empty_state(dim) if state is None else state
    count0 = state[2]
    emitted, start = [], 0
    for stop in cuts:
        where = f" (rows {start} to {stop - 1})" if public else ""
        result = ctx.call(FN, kv[start:stop], gate[start:stop], position_bias, gain, state, rate)
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            f"{FN}: the result is not a pair (entries, state)",
        )
        entries, state = result
        done_before, done = start // rate, stop // rate
        check.shape(entries, (done - done_before, dim), label=f"entries{where}")
        check.allclose(entries, want[done_before:done], **TOL, label=f"entries{where}", dtype=True)
        check.that(
            isinstance(state, tuple) and len(state) == 3,
            f"{FN}: the returned state is not a triple (buf_kv, buf_gate, count)",
        )
        buf_kv, buf_gate, count = state
        left = stop - done * rate
        for name, buf, rows in (("buf_kv", buf_kv, kv), ("buf_gate", buf_gate, gate)):
            check.that(
                isinstance(buf, np.ndarray), f"{name}{where} in the returned state is not a tensor"
            )
            check.shape(buf, (left, dim), label=f"{name}{where} in the returned state")
            check.allclose(
                buf, rows[stop - left : stop], **TOL, label=f"{name}{where} in the returned state"
            )
        check.that(
            type(count) is int,
            f"count{where} in the returned state is a {type(count).__name__}, not an int",
        )
        check.equal(count, count0 + done, label=f"count{where} in the returned state")
        emitted.append(entries)
        start = stop
    return np.concatenate(emitted), state


def check_visible(ctx, t, rate, expected):
    got = ctx.call(VIS, t, rate)
    check.that(type(got) is int, f"{VIS}: returned a {type(got).__name__}, not an int")
    check.equal(got, expected, label=f"{VIS}({t}, {rate})")


@case("the worked example: window of 2, three rows and then one", public=True)
def worked_example(ctx):
    ln3 = float(np.log(3.0))
    kv = np.array([[4, 2], [0, 6], [1, 1], [-3, 3]], np.float32)
    gate = np.array([[0, 0], [0, 0], [0, 0], [ln3, 0]], np.float32)
    position_bias = np.array([[ln3, 0], [0, 0]], np.float32)
    gain = np.array([1, 2], np.float32)
    got, state = stream(ctx, kv, gate, position_bias, gain, 2, [3, 4], public=True)
    expected = np.array([[0.848528, 2.262742], [-0.632455, 2.529822]], np.float32)
    check.allclose(got, expected, **TOL, label="entries")
    check.equal(state[2], 2, label="count after both calls")
    for t, expected_count in ((0, 0), (1, 1), (2, 1), (3, 2)):
        check_visible(ctx, t, 2, expected_count)


@case("14 rows, window of 4: one call, then calls of 3, 1, 9, and 1 rows", public=True)
def one_call_and_pieces(ctx):
    args = draw(ctx)
    stream(ctx, *args, RATE, WHOLE, public=True)
    stream(ctx, *args, RATE, PIECES, public=True)


@case("DeepseekV4HCACompressor (Hugging Face Transformers), window of 4", public=True)
def shipped_small(ctx):
    f = ctx.fixture("shipped.npz")
    args = tuple(f[f"w4_{name}"] for name in ("kv", "gate", "position_bias", "gain"))
    cuts = [int(c) for c in f["w4_cuts"]]
    got, state = stream(ctx, *args, int(f["w4_rate"]), cuts, public=True)
    check.allclose(got, f["w4_entries"], **TOL, label="entries against the library")
    check.allclose(state[0], f["w4_buffer_kv"], **TOL, label="buf_kv against the library's buffer")
    check.allclose(
        state[1], f["w4_buffer_gate"], **TOL, label="buf_gate against the library's buffer"
    )
    check.equal(state[2], int(f["w4_count"]), label="count against the library's entry count")


# The shipped window: a call of fewer rows than a window, a cut inside a window, a call
# of one row.
@case("DeepseekV4HCACompressor, the shipped window of 128: 300 rows in four calls", public=True)
def shipped_window(ctx):
    f = ctx.fixture("shipped.npz")
    args = tuple(f[f"w128_{name}"] for name in ("kv", "gate", "position_bias", "gain"))
    cuts = [int(c) for c in f["w128_cuts"]]
    got, state = stream(ctx, *args, int(f["w128_rate"]), cuts, public=True)
    check.allclose(got, f["w128_entries"], **TOL, label="entries against the library")
    check.allclose(
        state[0], f["w128_buffer_kv"], **TOL, label="buf_kv against the library's buffer"
    )
    check.equal(state[2], int(f["w128_count"]), label="count against the library's entry count")


# Rejects threshold_floor_t and threshold_rounds_up.
@case("visible_entries at the last position of a window and at the first of the next", public=True)
def visibility_at_window_edges(ctx):
    for rate in (4, 128):
        for w in (1, 2, 3):
            check_visible(ctx, w * rate - 2, rate, w - 1)  # one before the window's end
            check_visible(ctx, w * rate - 1, rate, w)  # the window's last position
            check_visible(ctx, w * rate, rate, w)  # the first position of the next window
    check_visible(ctx, 2**24 + 1, 2, 2**23 + 1)  # beyond exact float32 integers
    f = ctx.fixture("shipped.npz")  # counted from the library's mask, window of 4
    for t, expected in enumerate(f["w4_visible"]):
        check_visible(ctx, t, 4, int(expected))


@case("every way of cutting the stream gives the same entries", public=True)
def any_cutting(ctx):
    args = draw(ctx)
    for cuts in (WHOLE, PIECES, ROWS, INSIDE, TWO):
        stream(ctx, *args, RATE, cuts, public=True)


@case(
    "the tensors and the state passed in are not modified: a state is continued twice", public=True
)
def inputs_unchanged(ctx):
    kv, gate, position_bias, gain = draw(ctx)
    want = window_entries(kv, gate, position_bias, gain, RATE)
    fw = ctx.fw
    bias_t, gain_t = fw.tensor(position_bias), fw.tensor(gain)
    empty = (fw.tensor(np.zeros((0, D), np.float32)), fw.tensor(np.zeros((0, D), np.float32)), 0)
    kv_t, gate_t = fw.tensor(kv[:3]), fw.tensor(gate[:3])
    _, state = ctx.solution.compress(kv_t, gate_t, bias_t, gain_t, empty, RATE)
    for name, tensor, before in (("kv", kv_t, kv[:3]), ("gate", gate_t, gate[:3])):
        check.allclose(fw.numpy(tensor), before, rtol=0, atol=0, label=f"{name} after the call")
    before = [fw.numpy(state[0]).copy(), fw.numpy(state[1]).copy(), state[2]]
    for _ in range(2):  # the same continuation twice, from the same state
        entries, _ = ctx.solution.compress(
            fw.tensor(kv[3:4]), fw.tensor(gate[3:4]), bias_t, gain_t, state, RATE
        )
        check.allclose(fw.numpy(entries), want[:1], **TOL, label="entries of a call that continues")
        check.allclose(fw.numpy(state[0]), before[0], rtol=0, atol=0, label="buf_kv passed in")
        check.allclose(fw.numpy(state[1]), before[1], rtol=0, atol=0, label="buf_gate passed in")
        check.equal(state[2], before[2], label="count passed in")
    check.allclose(
        fw.numpy(bias_t), position_bias, rtol=0, atol=0, label="position_bias after the calls"
    )


@case("an entry depends only on its own window")
def window_independence(ctx):
    kv, gate, position_bias, gain = draw(ctx)
    base, _ = stream(ctx, kv, gate, position_bias, gain, RATE, PIECES)
    kv2, gate2 = kv.copy(), gate.copy()
    kv2[4:8] += ctx.randn(4, D)
    gate2[4:8] += ctx.randn(4, D)
    kv2[12:] += 5.0  # the unfinished window
    changed, _ = stream(ctx, kv2, gate2, position_bias, gain, RATE, PIECES)
    check.allclose(changed[[0, 2]], base[[0, 2]], **TOL, label="entries of the unchanged windows")
    check.that(
        float(np.abs(changed[1] - base[1]).max()) > 1e-3,
        "the entry of a window did not change although its rows did",
    )


# Rejects naive_softmax: exp(gate) overflows float32 above about 88.
@case("large gates")
def large_gates(ctx):
    kv, gate, position_bias, gain = draw(ctx, gate_scale=60.0)
    gate[1] += 150.0
    gate[9] -= 150.0
    gate = np.clip(gate, -390.0, 390.0)  # with the bias, inside the stated |z + b| <= 400
    stream(ctx, kv, gate, position_bias, gain, RATE, PIECES)


# Rejects norm_without_eps: at this scale the mean square of an entry is comparable
# with 1e-6; and an all-zero window must give a zero entry, not NaN.
@case("small values, and a window of zeros")
def small_values(ctx):
    kv, gate, position_bias, gain = draw(ctx)
    kv = kv * 1e-3
    stream(ctx, kv, gate, position_bias, gain, RATE, INSIDE)
    kv[4:8] = 0.0
    got, _ = stream(ctx, kv, gate, position_bias, gain, RATE, INSIDE)
    check.finite(got, label="entries")


@case("a window of one row, and a state that starts mid-stream")
def other_windows(ctx):
    kv, gate, position_bias, gain = draw(ctx, rate=1)
    stream(ctx, kv, gate, position_bias, gain, 1, WHOLE)
    stream(ctx, kv, gate, position_bias, gain, 1, ROWS[:3])
    # Rejects count_not_accumulated on a single call: the count does not start at 0.
    kv, gate, position_bias, gain = draw(ctx)
    resumed = (kv[12:14], gate[12:14], 3)  # rows 12 and 13 wait; three entries exist
    more_kv, more_gate = ctx.randn(9, D), ctx.randn(9, D)
    full_kv, full_gate = np.concatenate((kv, more_kv)), np.concatenate((gate, more_gate))
    want = window_entries(full_kv, full_gate, position_bias, gain, RATE)
    entries, state = ctx.call(FN, more_kv, more_gate, position_bias, gain, resumed, RATE)
    check.allclose(entries, want[3:5], **TOL, label="entries", dtype=True)
    check.equal(state[2], 5, label="count in the returned state")
    check.allclose(state[0], full_kv[20:], **TOL, label="buf_kv in the returned state")


@case("visible_entries for many positions and windows")
def visibility_sweep(ctx):
    for rate in (1, 2, 3, 4, 7, 128):
        for t in [*range(3 * rate + 2), 1000, 65535, 65536]:
            got = ctx.call(VIS, t, rate)
            check.that(type(got) is int, f"{VIS}: returned a {type(got).__name__}, not an int")
            complete = sum(1 for w in range(t // rate + 2) if (w + 1) * rate - 1 <= t)
            check.that(got == complete, f"{VIS}: got {got}, which is not the expected number")
    # Rejects float32_threshold: positions that float32 cannot represent exactly.
    for rate in (1, 2, 3, 128):
        for t in (2**24 + 1, 2**24 + 3, 2**25 + 5, 2**31 - 2, 2**31 - 1):
            got = ctx.call(VIS, t, rate)
            check.that(type(got) is int, f"{VIS}: returned a {type(got).__name__}, not an int")
            check.that(
                got == (t + 1) // rate, f"{VIS}: got {got}, which is not the expected number"
            )


@case("random streams, every cutting")
def random_sweep(ctx):
    def prop(rng):
        args = draw(ctx, gate_scale=float(rng.choice([0.1, 1.0, 8.0])))
        cuts = [WHOLE, PIECES, ROWS, INSIDE, TWO][int(rng.integers(0, 5))]
        stream(ctx, *args, RATE, cuts)

    check.forall(prop, trials=8)
