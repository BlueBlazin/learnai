"""Tests for entropy-bound-denoise-step.

Two oracles, neither of them this problem's reference: `step_ref`, the definition in
float64 numpy, and arrays computed by `_denoising_step` of `transformers` with its own
temperature schedule and `EntropyBoundSampler` (`fixtures/shipped.npz`, written by
`fixtures/make.py`).

Tokens and masks are compared exactly, so every input is built clear of float32
rounding (the statement's Constraints): `safe_uniforms` puts each `u_sample` in the
middle of a token's probability interval, `noise_uniforms` puts `u_noise * V` in the
middle of an integer interval, and `bound_accepting` puts the bound halfway between two
sums, between two positions whose entropies are equal by construction (identical rows)
or clearly apart. `step_ref` asserts all of it (an author's check, not the solution's).

Shapes are few (JAX compiles once per shape): 16 positions of 40 tokens, plus the
worked example (4 of 3), two cases of 256 positions of 8 tokens, and one of 4 positions
with DiffusionGemma's vocabulary of 262144.
"""

import numpy as np

from learnai.judge import case, check

TEMP, STEP = "step_temperature", "denoise_step"
L, V = 16, 40
T_MIN, T_MAX, N_MAX = 0.4, 0.8, 48  # the shipped schedule
ONE_BELOW = np.nextafter(np.float32(1.0), np.float32(0.0))  # the largest float32 below 1


def probabilities(logits, temperature):
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def entropies(p):
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)


def safe_uniforms(p, rng):
    """One uniform per row, in the middle of a token's interval at least 4e-3 wide."""
    cdf = np.cumsum(p, axis=-1)
    low = np.concatenate((np.zeros((p.shape[0], 1)), cdf[:, :-1]), axis=-1)
    out = np.empty(p.shape[0], dtype=np.float32)
    for i in range(p.shape[0]):
        token = min(int(np.searchsorted(cdf[i], rng.random(), side="right")), p.shape[1] - 1)
        if p[i, token] < 4e-3:
            token = int(p[i].argmax())
        out[i] = low[i, token] + p[i, token] * rng.uniform(0.3, 0.7)
    return out


def noise_uniforms(length, vocab, rng):
    tokens = rng.integers(0, vocab, length)
    return ((tokens + rng.uniform(0.25, 0.75, length)) / vocab).astype(np.float32)


def bound_accepting(h, count):
    """A bound that accepts exactly `count` positions: halfway between two sums."""
    ranked = np.sort(h)
    return float(ranked[: count - 1].sum() + ranked[count - 1] / 2)


def step_ref(logits, temperature, entropy_bound, u_sample, u_noise, min_gap=1e-3):
    """The definition, in float64: (canvas, accepted, argmax_canvas).

    Asserts that the input decides nothing by rounding. `min_gap` is the smallest
    allowed difference between the entropies on the two sides of the cut, unless their
    rows are identical.
    """
    vocab = logits.shape[-1]
    p = probabilities(logits, temperature)
    cdf = np.cumsum(p, axis=-1)
    threshold = u_sample.astype(np.float64)[:, None] * cdf[:, -1:]
    sampled = (cdf <= threshold).sum(axis=-1)
    exact = (u_sample == 0) | (u_sample == ONE_BELOW)
    clear = np.abs(cdf - threshold).min(axis=-1) >= 1e-3
    assert bool((clear | exact).all()), "test input: u_sample on a cumulative sum"

    h = entropies(p)
    order = np.argsort(h, kind="stable")
    before = np.cumsum(h[order]) - h[order]
    keep = before <= entropy_bound
    # A sum of exactly 0 (nothing before, or only zero entropies) is exact for everyone.
    near = np.abs(before - entropy_bound)[before > 0]
    assert near.size == 0 or float(near.min()) >= 1e-3, "test input: bound on a sum"
    count = int(keep.sum())
    if count < len(h):
        inside, outside = order[count - 1], order[count]
        tied = np.array_equal(logits[inside], logits[outside])
        assert tied or h[outside] - h[inside] >= min_gap, "test input: entropies at the cut"
    accepted = np.zeros(len(h), dtype=bool)
    accepted[order] = keep

    product = u_noise.astype(np.float64) * vocab
    noise = np.floor(product).astype(np.int64)
    off = np.abs(product - np.round(product))
    ends = (u_noise == 0) | (u_noise == ONE_BELOW)  # float32 and float64 agree at both
    assert bool(((off >= 1e-3) | ends)[~accepted].all()), "test input: u_noise * V"
    top2 = np.sort(logits, axis=-1)[:, -2:]
    assert vocab == 1 or float((top2[:, 1] - top2[:, 0]).min()) > 1e-4, "test input: argmax tie"
    return np.where(accepted, sampled, noise), accepted, logits.argmax(axis=-1)


def run(ctx, current, logits, temperature, entropy_bound, u_sample, u_noise, want=None, **ref):
    """Call denoise_step and compare the three results exactly."""
    if want is None:
        want = step_ref(logits, temperature, entropy_bound, u_sample, u_noise, **ref)
    got = ctx.call(STEP, current, logits, temperature, entropy_bound, u_sample, u_noise)
    check.that(
        isinstance(got, tuple) and len(got) == 3,
        f"{STEP}: the result is not a tuple (canvas, accepted, argmax_canvas)",
    )
    canvas, accepted, argmax_canvas = got
    check.that(
        accepted.dtype == np.bool_, f"accepted: got dtype {accepted.dtype}, which is not bool"
    )
    check.equal(accepted, want[1], label="accepted")
    for label, tokens in (("canvas", canvas), ("argmax_canvas", argmax_canvas)):
        check.that(
            tokens.dtype.kind in "iu", f"{label}: got dtype {tokens.dtype}, which is not an integer"
        )
    check.equal(argmax_canvas, want[2], label="argmax_canvas")
    check.equal(canvas, want[0], label="canvas")
    return got


def draw(ctx, temperature, length=L, vocab=V, low=0.1, high=4.0):
    """(current, logits, u_sample, u_noise): rows from nearly certain to nearly uniform."""
    sharp = ctx.rng.uniform(low, high, (length, 1)).astype(np.float32)
    logits = ctx.randn(length, vocab) * sharp
    best = logits.argmax(axis=-1)
    logits[np.arange(length), best] += np.float32(0.01)  # a unique largest logit per row
    p = probabilities(logits, temperature)
    return (
        ctx.randint(0, vocab, (length,)),
        logits,
        safe_uniforms(p, ctx.rng),
        noise_uniforms(length, vocab, ctx.rng),
    )


def draw_with_bound(ctx, temperature, count, **kwargs):
    """Inputs and a bound that accepts `count` positions with clear margins."""
    while True:
        current, logits, u_sample, u_noise = draw(ctx, temperature, **kwargs)
        ranked = np.sort(entropies(probabilities(logits, temperature)))
        if ranked[count - 1] >= 2e-3 and (
            count == len(ranked) or ranked[count] - ranked[count - 1] >= 2e-3
        ):
            bound = bound_accepting(ranked, count)
            return current, logits, temperature, bound, u_sample, u_noise


def temperature_of(ctx, n, n_max=N_MAX, t_min=T_MIN, t_max=T_MAX):
    got = ctx.call(TEMP, n, n_max, t_min, t_max)
    check.that(type(got) is float, f"{TEMP}: returned a {type(got).__name__}, not a Python float")
    want = t_min + (t_max - t_min) * n / n_max
    check.allclose(got, want, rtol=1e-9, atol=0, label=TEMP)
    return got


def shipped(ctx, i):
    f = ctx.fixture("shipped.npz")
    g = {
        name: f[f"r{i}_{name}"]
        for name in ("current", "logits", "n", "n_max", "t_min", "t_max", "entropy_bound")
    }
    temperature = temperature_of(
        ctx, int(g["n"]), int(g["n_max"]), float(g["t_min"]), float(g["t_max"])
    )
    check.allclose(temperature, float(f[f"r{i}_temperature"]), rtol=1e-6, atol=0, label=TEMP)
    want = tuple(f[f"r{i}_{name}"] for name in ("canvas", "accepted", "argmax"))
    return run(
        ctx,
        g["current"],
        g["logits"],
        temperature,
        float(g["entropy_bound"]),
        f[f"r{i}_u_sample"],
        f[f"r{i}_u_noise"],
        want=want,
    )


@case("the worked example: 4 positions, 3 tokens, temperature 0.5", public=True)
def worked_example(ctx):
    check.allclose(temperature_of(ctx, 12), 0.5, rtol=1e-9, atol=0, label=TEMP)
    canvas, accepted, argmax_canvas = shipped(ctx, 0)
    check.equal(accepted, [False, True, False, True], label="accepted")
    check.equal(canvas, [2, 0, 1, 2], label="canvas")
    check.equal(argmax_canvas, [2, 0, 1, 0], label="argmax_canvas")
    canvas, accepted, _ = shipped(ctx, 1)  # entropy_bound = 0.3
    check.equal(accepted, [False, True, False, False], label="accepted with entropy_bound = 0.3")
    check.equal(canvas, [2, 0, 1, 1], label="canvas with entropy_bound = 0.3")


@case("step_temperature on the shipped schedule: 0.8 down to 0.4 over 48 steps", public=True)
def temperature_schedule(ctx):
    for n, want in ((48, 0.8), (36, 0.7), (24, 0.6), (1, 0.4 + 0.4 / 48)):
        check.allclose(temperature_of(ctx, n), want, rtol=1e-9, atol=0, label=f"{TEMP} for n = {n}")
    temperature_of(ctx, 3, 8, 0.25, 1.5)
    temperature_of(ctx, 5, 5, 1.0, 1.0)


@case("whole steps against _denoising_step of transformers, 16 positions", public=True)
def shipped_steps(ctx):
    for i in range(2, 9):
        shipped(ctx, i)


# Rejects ties_to_highest_index: five identical rows, and a bound that admits two of them.
@case("equal entropies: identical rows are accepted in index order", public=True)
def tied_entropies(ctx):
    temperature = 0.65
    current, logits, _, u_noise = draw(ctx, temperature, low=0.1, high=0.4)
    same = [1, 3, 4, 6, 9]
    logits[same] = ctx.randn(V) * np.float32(2.0)
    lower = [0, 12]
    logits[lower] = ctx.randn(2, V) * np.float32(6.0)
    p = probabilities(logits, temperature)
    h = entropies(p)
    assert h[lower].max() < h[1] - 0.05 and np.delete(h, same + lower).min() > h[1] + 0.05
    u_sample = safe_uniforms(p, ctx.rng)
    for admitted in (2, 4, 1):
        bound = float(h[lower].sum() + (admitted - 0.5) * h[1])
        _, accepted, _ = run(ctx, current, logits, temperature, bound, u_sample, u_noise)
        check.equal(
            np.flatnonzero(accepted),
            sorted(lower + same[:admitted]),
            label="accepted positions",
        )


# Rejects unstable_sort. A sort that is not stable returns equal keys in some other
# order than by index once the array is long (PyTorch's default sort is stable only
# for short inputs). 100 identical rows among 256 positions, and bounds that admit 1,
# 37, and 99 of them: the accepted ones must be the lowest 1, 37, and 99 indices, which
# an order other than by index cannot produce for all three.
@case("256 positions, 100 of them identical rows: accepted in index order", public=True)
def tied_entropies_long(ctx):
    length, vocab, temperature = 256, 8, 0.8
    while True:
        current, logits, _, u_noise = draw(
            ctx, temperature, length=length, vocab=vocab, low=0.05, high=0.3
        )
        places = ctx.rng.permutation(length)
        same, lower = np.sort(places[:100]), np.sort(places[100:120])
        logits[same] = ctx.randn(vocab) * np.float32(1.2)
        logits[lower] = ctx.randn(20, vocab) * np.float32(5.0)
        best = logits.argmax(axis=-1)
        logits[np.arange(length), best] += np.float32(0.01)
        p = probabilities(logits, temperature)
        h = entropies(p)
        rest = np.delete(h, np.concatenate((same, lower)))
        if h[lower].max() < h[same[0]] - 0.05 and rest.min() > h[same[0]] + 0.05:
            break
    u_sample = safe_uniforms(p, ctx.rng)
    for admitted in (37, 1, 99):
        bound = float(h[lower].sum() + (admitted - 0.5) * h[same[0]])
        _, accepted, _ = run(ctx, current, logits, temperature, bound, u_sample, u_noise)
        check.equal(
            np.flatnonzero(accepted),
            np.sort(np.concatenate((lower, same[:admitted]))),
            label=f"accepted positions when the bound admits {admitted} of the identical rows",
        )


@case(
    "the lowest-entropy position is always accepted; a larger bound accepts a superset", public=True
)
def monotone_in_the_bound(ctx):
    def prop(rng):
        temperature = float(rng.choice([0.4, 0.65, 0.8]))
        while True:
            current, logits, u_sample, u_noise = draw(ctx, temperature)
            ranked = np.sort(entropies(probabilities(logits, temperature)))
            if ranked[0] >= 2e-3 and ranked[1] - ranked[0] >= 1e-2:
                break
        lowest = int(entropies(probabilities(logits, temperature)).argmin())
        previous = np.zeros(L, dtype=bool)
        for bound in (1e-6, 0.02, 0.3, 1.0, 2.5, 6.0, 15.0, 1e3):
            got = ctx.call(STEP, current, logits, temperature, bound, u_sample, u_noise)
            check.that(
                isinstance(got, tuple) and len(got) == 3,
                f"{STEP}: the result is not a tuple (canvas, accepted, argmax_canvas)",
            )
            accepted = np.asarray(got[1])
            check.shape(accepted, (L,), label="accepted")
            check.that(
                bool(accepted[lowest]),
                "accepted: the position with the lowest entropy was not accepted",
                f"entropy_bound = {bound}, position {lowest}",
            )
            check.that(
                bool((accepted | ~previous).all()),
                "accepted: a position accepted with a smaller bound is rejected with a larger one",
                f"entropy_bound = {bound}",
            )
            previous = accepted.astype(bool)
        check.that(bool(previous.all()), "accepted: a bound of 1000 did not accept every position")

    check.forall(prop, trials=3)


@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    current, logits, temperature, bound, u_sample, u_noise = draw_with_bound(ctx, 0.6, 6)
    held = {"current": current, "logits": logits, "u_sample": u_sample, "u_noise": u_noise}
    t = {name: ctx.fw.tensor(a) for name, a in held.items()}
    ctx.solution.denoise_step(
        t["current"], t["logits"], temperature, bound, t["u_sample"], t["u_noise"]
    )
    for name, before in held.items():
        check.allclose(
            ctx.fw.numpy(t[name]), before, rtol=0, atol=0, label=f"{name} after the call"
        )


@case("random canvases at several temperatures and acceptance counts")
def random_canvases(ctx):
    for temperature, count in ((0.8, 2), (0.55, 9), (0.4, 14), (1.7, 5), (0.4, 16), (0.7, 1)):
        run(ctx, *draw_with_bound(ctx, temperature, count))


# Rejects naive_entropy_nan: after the division by a low temperature, logit gaps of
# several hundred make probabilities exactly 0 and 1 in float32.
@case("a canvas that is certain everywhere, and one that is certain in places")
def certain_positions(ctx):
    temperature = 0.4
    current, logits, _, u_noise = draw(ctx, temperature, low=0.5, high=2.0)
    spiked = logits.copy()
    spiked[np.arange(L), ctx.randint(0, V, (L,))] += 400.0
    u_sample = safe_uniforms(probabilities(spiked, temperature), ctx.rng)
    run(ctx, current, spiked, temperature, 0.1, u_sample, u_noise)
    half = logits.copy()
    half[::2] = spiked[::2]
    p = probabilities(half, temperature)
    ranked = np.sort(entropies(p))
    bound = float(ranked[8] + ranked[9] / 2)  # the eight certain positions and two more
    run(ctx, current, half, temperature, bound, safe_uniforms(p, ctx.rng), u_noise)


# Rejects offset_tie_break: two entropies 2e-5 apart, the smaller one 255 positions later.
@case("256 positions: entropies that differ in the fifth digit")
def near_equal_entropies(ctx):
    length, vocab = 256, 8
    current, logits, _, u_noise = draw(ctx, 1.0, length=length, vocab=vocab, low=0.2, high=0.8)
    logits[255] = 0.0
    logits[0] = 0.0
    logits[255, 0] = 4.5
    logits[0, 0] = 4.5 - 8e-5
    p = probabilities(logits, 1.0)
    h = entropies(p)
    assert 1e-5 <= h[0] - h[255] <= 4e-5 and np.sort(h)[2] > h[0] + 0.1
    u_sample = safe_uniforms(p, ctx.rng)
    for bound, want in ((h[255] / 2, [255]), (1.5 * h[255], [0, 255])):
        _, accepted, _ = run(
            ctx, current, logits, 1.0, float(bound), u_sample, u_noise, min_gap=1e-5
        )
        check.equal(np.flatnonzero(accepted), want, label="accepted positions")


# The vocabulary of DiffusionGemma. Three identical rows with entropy near 6.9 and one
# sharper row: ties at a large entropy, and noise tokens over the whole vocabulary,
# including the largest float32 below 1.
@case("4 positions over a vocabulary of 262144")
def full_vocabulary(ctx):
    vocab = 262144
    row = ctx.randn(vocab, scale=0.01)
    row[77] = np.log(vocab - 1.0)  # about half of the mass on token 77
    sharp = row.copy()
    sharp[77] = 20.0
    logits = np.stack((row, sharp, row, row))
    p = probabilities(logits, 1.0)
    h = entropies(p)
    assert h[0] > 6.5 and h[1] < 0.02
    current = ctx.randint(0, vocab, (4,))
    u_sample = np.array([0.25, 0.5, 0.3, 0.2], dtype=np.float32)
    u_noise = np.array([ONE_BELOW, 0.3, 0.0, (123456 + 0.5) / vocab], dtype=np.float32)
    canvas, accepted, _ = run(
        ctx, current, logits, 1.0, float(h[1] + 1.5 * h[0]), u_sample, u_noise
    )
    check.equal(accepted, [True, True, True, False], label="accepted")
    check.equal(canvas, [77, 77, 77, 123456], label="canvas")
    canvas, accepted, _ = run(
        ctx, current, logits, 1.0, float(h[1] + 0.5 * h[0]), u_sample, u_noise
    )
    check.equal(accepted, [True, True, False, False], label="accepted")
    canvas, _, _ = run(ctx, current, logits, 1.0, float(h[1] / 2), u_sample, u_noise)
    check.equal(canvas, [vocab - 1, 77, 0, 123456], label="canvas")


# Rejects sample_not_strict and noise_rounds. Rows whose arithmetic is exact: tokens of
# probability exactly 0 before and after the mass, with u_sample = 0 and the largest
# float32 below 1; and u_noise = 0, just below 1, and three quarters into an interval.
@case("the ends of the uniform range")
def uniform_range_ends(ctx):
    current, logits, _, _ = draw(ctx, 1.0, low=0.5, high=1.5)
    logits[:, :3] -= 1000.0
    logits[:, -3:] -= 1000.0
    p = probabilities(logits, 1.0)
    assert bool((p[:, :3] == 0).all() and (p[:, -3:] == 0).all())
    assert float(p[:, 3].min()) > 1e-5 and float(p[:, -4].min()) > 1e-5
    u_sample = safe_uniforms(p, ctx.rng)
    u_sample[::3] = 0.0
    u_sample[1::3] = ONE_BELOW
    tokens = ctx.randint(0, V, (L,))
    u_noise = ((tokens + 0.75) / V).astype(np.float32)
    u_noise[::4] = 0.0
    u_noise[1::4] = ONE_BELOW
    run(ctx, current, logits, 1.0, 1e3, u_sample, u_noise)
    ranked = np.sort(entropies(p))
    assert ranked[0] >= 2e-3
    run(ctx, current, logits, 1.0, float(ranked[0] / 2), u_sample, u_noise)


@case("a canvas of one position")
def one_position(ctx):
    for bound in (1e-6, 3.0):
        current, logits, u_sample, u_noise = draw(ctx, 0.5, length=1, low=0.5, high=1.0)
        _, accepted, _ = run(ctx, current, logits, 0.5, bound, u_sample, u_noise)
        check.equal(accepted, [True], label="accepted")
