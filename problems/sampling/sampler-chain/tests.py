"""Tests for sampler-chain.

Numeric cases compare `chain_logits` (and the tokens of `sample_chain`) with the
oracle. Which tokens survive depends on comparisons (the k-th largest logit, a
partial sum against top_p, a probability ratio against min_p, u against the cumulative
sums), and float32 rounding must never decide one of them. So `chain64` below runs the
whole chain in float64 with plain numpy, independently of the reference, and reports
how close the closest comparison was; inputs are redrawn (random cases) or rejected
(fixed cases) when that margin is below MARGIN. `chain64` also supplies the exact
target distribution of the statistical cases.

There are three statistical checks, each with the default delta of 1e-10, which keeps
the false-failure rate of a correct solution below 1e-9.
"""

import numpy as np

from learnai.judge import case, check

MARGIN = 1e-3
OFF = {
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "temperature": 1.0,
}


def config(**changes):
    unknown = set(changes) - set(OFF)
    if unknown:
        raise RuntimeError(f"authoring error: unknown config keys {unknown}")
    return {**OFF, **changes}


def f32(values):
    return np.array(values, dtype=np.float32)


def softmax64(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def chain64(logits, cfg, history):
    """The chain of the statement in float64. Returns (final logits, smallest margin)."""
    x = np.asarray(logits, dtype=np.float64).copy()
    vocab = x.shape[1]
    margin = np.inf
    counts = np.stack([np.bincount(row, minlength=vocab) for row in np.asarray(history)])
    seen = counts > 0
    penalty = cfg["repetition_penalty"]
    x = np.where(seen, np.where(x > 0, x / penalty, x * penalty), x)
    x = x - counts * cfg["frequency_penalty"] - seen * cfg["presence_penalty"]

    k = cfg["top_k"]
    if 0 < k < vocab:
        ranked = -np.sort(-x, axis=-1)
        finite = np.isfinite(ranked[:, k])
        if finite.any():
            margin = min(margin, float((ranked[finite, k - 1] - ranked[finite, k]).min()))
        order = np.argsort(-x, axis=-1, kind="stable")
        rank = np.argsort(order, axis=-1, kind="stable")
        x = np.where(rank < k, x, -np.inf)

    p = cfg["top_p"]
    if p < 1.0:
        probs = softmax64(x)
        order = np.argsort(-probs, axis=-1, kind="stable")
        ranked = np.take_along_axis(probs, order, axis=-1)
        before = np.cumsum(ranked, axis=-1) - ranked
        # Only boundaries between tokens that are still alive can change the outcome.
        alive = ranked[:, 1:] > 0
        if alive.any():
            margin = min(margin, float(np.abs(before[:, 1:] - p)[alive].min()))
        before = np.take_along_axis(before, np.argsort(order, axis=-1, kind="stable"), axis=-1)
        x = np.where(before < p, x, -np.inf)

    min_p = cfg["min_p"]
    if min_p > 0.0:
        probs = softmax64(x)
        ratio = probs / probs.max(axis=-1, keepdims=True)
        alive = ratio > 0
        margin = min(margin, float(np.abs(ratio / min_p - 1.0)[alive].min()))
        x = np.where(ratio >= min_p, x, -np.inf)

    return x / cfg["temperature"], margin


def require_clear(logits, cfg, history):
    final, margin = chain64(logits, cfg, history)
    if margin < MARGIN:  # an authoring error, not a solution failure
        raise RuntimeError(f"ambiguous test input: a comparison is within {margin:.2e}")
    return final


def clear_logits(ctx, draw, cfg, history):
    """Draw logits with `draw()` until no comparison of the chain is close."""
    while True:
        logits = draw()
        if chain64(logits, cfg, history)[1] >= MARGIN:
            return logits


def check_logits(ctx, logits, cfg, history):
    """Compare chain_logits with the oracle, after checking the input is unambiguous."""
    logits, history = f32(logits), np.asarray(history)
    final = require_clear(logits, cfg, history)
    got = ctx.check("chain_logits", logits, cfg, history)
    # The reference itself is checked against the float64 chain: same survivors.
    if not np.array_equal(np.isfinite(ctx.expected("chain_logits", logits, cfg, history)),
                          np.isfinite(final)):  # fmt: skip
        raise RuntimeError("authoring error: the reference and the float64 chain disagree")
    return got, final


def clear_uniforms(ctx, final, n=None):
    """One u per row, at least 1e-4 away (in probability) from every CDF boundary."""
    cdf = np.cumsum(softmax64(final), axis=-1)
    u = ctx.uniform(len(final))
    while True:
        close = np.abs(cdf - u[:, None].astype(np.float64)).min(axis=-1) < 1e-4
        if not close.any():
            return u
        u[close] = ctx.uniform(int(close.sum()))


def check_tokens(ctx, logits, cfg, history):
    """chain_logits against the oracle, then sample_chain's tokens against the oracle."""
    logits, history = f32(logits), np.asarray(history)
    _, final = check_logits(ctx, logits, cfg, history)
    u = clear_uniforms(ctx, final)
    tokens = np.asarray(ctx.call("sample_chain", logits, cfg, history, u))
    check.shape(tokens, (len(logits),), label="sample_chain")
    check.that(
        tokens.dtype.kind in "iu",
        f"sample_chain: expected an integer tensor of token ids, got dtype {tokens.dtype}",
    )
    check.equal(tokens, ctx.expected("sample_chain", logits, cfg, history, u), label="tokens")


def check_samples(ctx, logits, cfg, history, n=4000):
    """n independent draws from one row follow the chain's exact distribution."""
    logits, history = f32(logits)[None], np.asarray(history)[None]
    target = softmax64(require_clear(logits, cfg, history))[0]
    u = ctx.rng.random(n, dtype=np.float32)
    tokens = np.asarray(
        ctx.call("sample_chain", np.tile(logits, (n, 1)), cfg, np.tile(history, (n, 1)), u)
    )
    check.shape(tokens, (n,), label="sample_chain")
    check.that(
        bool(((tokens >= 0) & (tokens < logits.shape[1])).all()),
        "sample_chain: a returned token id is outside [0, V)",
    )
    check.distribution(tokens, target, label="sampled token")


# The worked example of the statement.
EXAMPLE_LOGITS = [[3.0, 2.5, 2.0, 1.0, 0.0, -1.0]]
EXAMPLE_HISTORY = [[0, 0, 5]]
EXAMPLE_CFG = config(
    repetition_penalty=1.2,
    frequency_penalty=0.5,
    presence_penalty=0.25,
    top_k=4,
    top_p=0.8,
    min_p=0.3,
    temperature=2.0,
)


@case("the worked example: every stage removes or changes something", public=True)
def worked_example(ctx):
    logits, history = f32(EXAMPLE_LOGITS), np.array(EXAMPLE_HISTORY)
    require_clear(logits, EXAMPLE_CFG, history)
    got = ctx.call("chain_logits", logits, EXAMPLE_CFG, history)
    inf = np.inf
    check.allclose(
        got, f32([[-inf, 1.25, 1.0, -inf, -inf, -inf]]), label="chain_logits", dtype=True
    )
    for u, token in ((0.3, 1), (0.7, 2)):
        tokens = ctx.call("sample_chain", logits, EXAMPLE_CFG, history, f32([u]))
        check.equal(tokens, [token], label=f"token for u = {u}")


@case("one stage at a time: each of the five stages alone, the others switched off", public=True)
def single_stages(ctx):
    history = ctx.randint(0, 10, (4, 3))
    for changes in (
        {"repetition_penalty": 1.3},
        {"frequency_penalty": 0.4, "presence_penalty": 0.7},
        {"top_k": 3},
        {"top_p": 0.7},
        {"min_p": 0.2},
        {"temperature": 0.5},
    ):
        cfg = config(**changes)
        logits = clear_logits(ctx, lambda: ctx.randn(4, 10, scale=2.0), cfg, history)
        check_logits(ctx, logits, cfg, history)


@case("temperature 2.0 and 0.5 with top-p and min-p: the filters see temperature 1", public=True)
def temperature_last(ctx):
    # Catches dividing by the temperature before the filters, or between them.
    history = ctx.randint(0, 10, (4, 3))
    for temperature in (2.0, 0.5):
        for changes in ({"top_p": 0.85}, {"min_p": 0.15}, {"top_p": 0.9, "min_p": 0.1}):
            cfg = config(temperature=temperature, **changes)
            logits = clear_logits(ctx, lambda: ctx.randn(4, 10, scale=1.5), cfg, history)
            check_logits(ctx, logits, cfg, history)


@case("4000 draws at temperature 1.6 follow the chain's distribution", public=True)
def samples_hot(ctx):
    # With the temperature applied first, the flatter distribution has a larger
    # nucleus, and tokens that the chain removes get sampled.
    logits = [2.0, 1.0, 3.0, 0.5, 2.5, -1.0, 1.5, 0.0]
    cfg = config(top_p=0.8, temperature=1.6, presence_penalty=0.5)
    check_samples(ctx, logits, cfg, [5, 2, 2])


@case("penalties: repeated tokens with positive, zero, and negative logits, and counts above 1")
def penalties(ctx):
    logits = [[2.0, -1.0, 0.0, 0.5, -3.0, 4.0], [-2.0, 3.0, 1.5, -0.25, 0.0, 0.75]]
    history = [[5, 1, 5, 2], [0, 0, 0, 4]]
    for changes in (
        {"repetition_penalty": 1.5},
        {"frequency_penalty": 0.3},
        {"presence_penalty": 0.6},
        {"repetition_penalty": 1.2, "frequency_penalty": 0.25, "presence_penalty": 0.5},
    ):
        check_logits(ctx, logits, config(**changes), history)


@case("penalties come first: a penalized token loses its place in the top-k and the nucleus")
def penalties_before_filters(ctx):
    # Token 0 has the largest raw logit and was generated three times. After the
    # penalties it is ranked fourth, so top_k = 3 (or the nucleus) must drop it.
    logits = [[4.0, 3.5, 3.0, 2.0, 1.0, 0.0], [1.0, 4.0, 0.5, 3.2, 2.6, -1.0]]
    history = [[0, 0, 0, 4], [1, 1, 1, 3]]
    for changes in ({"top_k": 3}, {"top_p": 0.75}, {"min_p": 0.3}, {"top_k": 4, "top_p": 0.9}):
        cfg = config(frequency_penalty=0.6, presence_penalty=0.4, **changes)
        check_tokens(ctx, logits, cfg, history)


@case("top-p after top-k: the nucleus is measured on the renormalized survivors")
def top_p_after_top_k(ctx):
    # Probabilities 0.3, 0.25, 0.2, 0.12, 0.08, 0.05. top_k = 3 leaves 0.4, 0.333,
    # 0.267, and top_p = 0.7 then keeps two tokens. On the unrenormalized masses 0.3
    # and 0.55, a third token would still be inside 0.7.
    logits = np.log(
        f32([[0.3, 0.25, 0.2, 0.12, 0.08, 0.05], [0.08, 0.2, 0.12, 0.3, 0.05, 0.25]])
    )
    history = [[0, 0, 0, 0], [0, 0, 0, 0]]
    check_logits(ctx, logits, config(top_k=3, top_p=0.7), history)
    check_logits(ctx, logits, config(top_k=4, top_p=0.8, temperature=0.7), history)
    check_logits(ctx, logits, config(top_k=2, top_p=0.5), history)


@case("everything switched off returns the logits unchanged, also with a far tail")
def all_off(ctx):
    # Logits 40 apart: the tail's probability is below float32 resolution. top_p = 1.0
    # means "no top-p stage", so the tail must stay finite.
    logits = f32([[0.0, 40.0, -3.0, 39.0, -60.0, 1.0], [5.0, -80.0, 4.0, 30.0, -2.0, 0.0]])
    history = np.array([[1, 3, 3, 1], [0, 0, 0, 0]])
    got = ctx.call("chain_logits", logits, config(), history)
    check.allclose(got, logits, label="chain_logits", dtype=True)
    check_logits(ctx, logits, config(top_k=6, temperature=4.0), history)
    check_logits(ctx, logits, config(top_k=50, min_p=0.01), history)


@case("masked logits (-inf) stay removed through every stage")
def masked_logits(ctx):
    history = np.array([[2, 2, 7], [0, 5, 5], [1, 1, 1], [3, 8, 9]])
    cfg = config(
        repetition_penalty=1.3, frequency_penalty=0.2, top_k=5, top_p=0.9, min_p=0.05,
        temperature=0.8,
    )  # fmt: skip

    def draw():
        logits = ctx.randn(4, 10, scale=2.0)
        logits[0, 2] = logits[0, 4] = logits[1, 0] = -np.inf
        logits[2, 2:] = -np.inf  # two live tokens
        return logits

    check_tokens(ctx, clear_logits(ctx, draw, cfg, history), cfg, history)


@case("4000 draws at temperature 0.6 follow the chain's distribution")
def samples_cold(ctx):
    # With the temperature applied first, the sharper distribution has a smaller
    # nucleus and min-p removes more, so tokens the chain keeps are never sampled.
    logits = [1.0, 2.2, 0.4, 1.8, 2.6, 1.4, -0.5, 0.9]
    cfg = config(top_p=0.9, min_p=0.12, temperature=0.6, repetition_penalty=1.15)
    check_samples(ctx, logits, cfg, [4, 4, 1])


@case("4000 draws through the full chain with top-k: the temperature shapes the survivors")
def samples_full_chain(ctx):
    # Catches sampling from the filtered distribution without the temperature.
    logits = [0.5, 3.0, 2.0, 1.0, 2.5, 0.0, 1.5, 2.8]
    cfg = config(top_k=5, top_p=0.95, min_p=0.05, temperature=2.5, frequency_penalty=0.3)
    check_samples(ctx, logits, cfg, [7, 7, 3])


@case("the tokens of a random batch: each row has its own history and its own u")
def random_tokens(ctx):
    history = ctx.randint(0, 10, (4, 3))
    cfg = config(
        repetition_penalty=1.1, presence_penalty=0.3, top_k=8, top_p=0.92, min_p=0.04,
        temperature=1.3,
    )  # fmt: skip
    for _ in range(3):
        logits = clear_logits(ctx, lambda: ctx.randn(4, 10, scale=2.0), cfg, history)
        check_tokens(ctx, logits, cfg, history)


@case("random configurations: a square batch, a single row, a history of one token")
def random_configs(ctx):
    # Few distinct shapes, for JAX.
    for shape, steps in (((6, 6), 1), ((1, 6), 3), ((4, 10), 3)):
        for _ in range(2 if shape[0] == 1 else 4):
            rng = ctx.rng
            cfg = config(
                repetition_penalty=round(float(rng.uniform(1.0, 1.5)), 2),
                frequency_penalty=round(float(rng.uniform(0.0, 0.5)), 2),
                presence_penalty=round(float(rng.uniform(0.0, 0.5)), 2),
                top_k=int(rng.integers(0, shape[1] + 3)),
                top_p=float(rng.choice([0.5, 0.8, 0.95, 1.0])),
                min_p=float(rng.choice([0.0, 0.02, 0.1, 0.3])),
                temperature=round(float(rng.uniform(0.3, 2.5)), 2),
            )
            history = ctx.randint(0, shape[1], (shape[0], steps))
            scale = float(rng.uniform(0.5, 4.0))

            def draw(shape=shape, scale=scale):
                return ctx.randn(*shape, scale=scale)

            logits = clear_logits(ctx, draw, cfg, history)
            check_tokens(ctx, logits, cfg, history)
