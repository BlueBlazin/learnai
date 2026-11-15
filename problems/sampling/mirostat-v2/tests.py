"""Tests for mirostat-v2.

A step makes two float32 comparisons whose outcome is discrete: `surprise <= mu`
(which tokens are kept) and `cdf > u * total` (which kept token is drawn). Frameworks
round a log-softmax and a cumulative sum differently in the last bits, so `model`, a
float64 implementation of the statement kept in this file, also reports how close an
input comes to either boundary, and every numeric case is drawn (or redrawn) until
both margins hold. Tokens are then compared exactly and `mu` within float32 noise.

The long-run cases feed the solution its own `mu` for hundreds of steps and compare
nothing with the oracle: they check each step against the statement with a tolerance,
and then what the sampler is for, the mean surprise of the output.
"""

import numpy as np

from whetstone.judge import case, check

FN = "mirostat_step"
INF = np.inf
TRUNC_MARGIN = 2e-4  # bits between any token's surprise and mu
CDF_MARGIN = 2e-5  # between any cumulative sum and the threshold, relative to the kept mass
V_SMALL, V_LARGE = 12, 200


def f32(values):
    return np.array(values, dtype=np.float32)


def surprise_bits(logits):
    """Surprise of every token in bits, float64; +inf for masked tokens."""
    x = np.asarray(logits, dtype=np.float64)
    peak = x.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_p = x - peak - np.log(np.exp(x - peak).sum(axis=-1, keepdims=True))
    return -log_p / np.log(2.0)


def model(logits, mu, tau, eta, u):
    """The statement in float64. Returns (tokens, new_mu, ok): ok[b] is False when row b
    is closer to a truncation or sampling boundary than float32 can be trusted with."""
    s = surprise_bits(logits)
    mu = np.asarray(mu, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    rows = np.arange(len(s))
    top = np.asarray(logits).argmax(axis=-1)
    keep = s <= mu[:, None]
    keep[rows, top] = True
    kept = np.where(keep, np.exp2(-s), 0.0)
    cdf = kept.cumsum(axis=-1)
    total = cdf[:, -1:]
    tokens = (cdf <= u[:, None] * total).sum(axis=-1)
    new_mu = mu - eta * (s[rows, tokens] - tau)
    gap = np.abs(s - mu[:, None])
    gap[rows, top] = INF  # the most likely token is kept on either side of the limit
    trunc_ok = gap.min(axis=-1) > TRUNC_MARGIN
    cdf_ok = (np.abs(cdf / total - u[:, None])).min(axis=-1) > CDF_MARGIN
    return tokens, new_mu, trunc_ok & cdf_ok


def call(ctx, logits, mu, tau, eta, u):
    """Call the solution; check the types and shapes of the pair it returns."""
    result = ctx.call(FN, logits, mu, tau, eta, u)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{FN}: expected a pair (token, new_mu)",
    )
    tokens, new_mu = np.asarray(result[0]), np.asarray(result[1])
    check.shape(tokens, (len(logits),), label="token")
    check.shape(new_mu, (len(logits),), label="new mu")
    check.that(
        tokens.dtype.kind in "iu",
        f"{FN}: expected an integer tensor of token indices, got dtype {tokens.dtype}",
    )
    check.that(
        new_mu.dtype == np.float32,
        f"{FN}: the new mu has dtype {new_mu.dtype}, which is not the expected dtype",
        "expected float32",
    )
    check.that(
        bool(((tokens >= 0) & (tokens < logits.shape[1])).all()),
        f"{FN}: a returned token is outside [0, V)",
        f"tokens = {tokens}",
    )
    return tokens.astype(np.int64), new_mu


def check_step(ctx, logits, mu, tau, eta, u):
    logits, mu, u = f32(logits), f32(mu), f32(u)
    want_tokens, want_mu, ok = model(logits, mu, tau, eta, u)
    if not ok.all():  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: a row is within the margin of a boundary")
    tokens, new_mu = call(ctx, logits, mu, tau, eta, u)
    detail = (
        f"logits =\n{logits}\nmu = {mu}\ntau = {tau}, eta = {eta}\nu = {u}\n"
        f"expected tokens {want_tokens}, got {tokens}\nexpected mu {want_mu}, got {new_mu}"
    )
    check.that(
        bool((tokens == want_tokens).all()),
        f"token: {int((tokens != want_tokens).sum())} of {len(tokens)} rows drew a token that "
        "is not the expected one",
        detail,
    )
    check.allclose(new_mu, want_mu, rtol=1e-4, atol=1e-4, label="new mu")
    return tokens, new_mu


def draw_inputs(ctx, logits, mu_low, mu_high):
    """Random mu and u for every row, redrawn until no row is near a boundary."""
    batch = len(logits)
    mu = ctx.uniform(batch, low=mu_low, high=mu_high)
    u = ctx.rng.random(batch, dtype=np.float32)
    while True:
        _, _, ok = model(logits, mu, 3.0, 0.1, u)
        if ok.all():
            return mu, u
        bad = ~ok
        mu[bad] = ctx.uniform(int(bad.sum()), low=mu_low, high=mu_high)
        u[bad] = ctx.rng.random(int(bad.sum()), dtype=np.float32)


def zipf_logits(ctx, vocab, exponent=1.1):
    """A Zipf-like next-token distribution in a random token order, as logits."""
    return ctx.rng.permutation(-exponent * np.log(np.arange(1, vocab + 1))).astype(np.float32)


EXAMPLE_ROW = np.log(f32([2, 8, 1, 4, 1]))  # probabilities 1/8, 1/2, 1/16, 1/4, 1/16


@case(
    "the worked example: everything kept, a truncated row, nothing within mu, a masked token",
    public=True,
)
def worked_example(ctx):
    with np.errstate(divide="ignore"):
        masked = np.log(f32([2, 8, 0, 4, 2]))
    logits = np.stack([EXAMPLE_ROW, EXAMPLE_ROW, EXAMPLE_ROW, masked])
    mu = f32([6.0, 2.5, 0.5, 3.5])
    u = f32([0.9, 0.7, 0.99, 0.95])
    tokens, new_mu = check_step(ctx, logits, mu, 3.0, 0.5, u)
    check.equal(tokens, [3, 3, 1, 4], label="token")
    check.allclose(new_mu, f32([6.5, 3.0, 1.5, 3.5]), atol=1e-4, label="new mu")


@case("a random batch: every row has its own mu and its own u", public=True)
def random_batch(ctx):
    # Catches a limit shared across rows, truncation at tau, sampling in sorted order,
    # and a missing renormalization.
    logits = ctx.randn(8, V_SMALL, scale=2.0)
    for tau, eta in ((3.0, 0.1), (5.0, 0.4)):
        mu, u = draw_inputs(ctx, logits, 0.5, 7.0)
        check_step(ctx, logits, mu, tau, eta, u)


@case("mu below every surprise: only the most likely token remains, and mu rises", public=True)
def nothing_within_mu(ctx):
    logits = ctx.randn(8, V_SMALL, scale=1.0)
    mu = ctx.uniform(8, low=-2.0, high=0.3)  # mu can go negative after a run of surprises
    mu[0] = 0.0
    u = ctx.rng.random(8, dtype=np.float32)
    tokens, new_mu = check_step(ctx, logits, mu, 4.0, 0.2, u)
    check.equal(tokens, logits.argmax(axis=-1), label="token")
    check.that(bool((new_mu > mu).all()), "new mu: did not rise after a token less surprising "
               "than tau")  # fmt: skip


@case("twenty steps of one sequence, feeding mu back: tokens and mu follow the reference",
      public=True)  # fmt: skip
def short_trace(ctx):
    # Catches natural logarithms, a flipped sign, and surprise measured after
    # renormalization: each gives a different mu after the first step.
    tau, eta, steps, batch = 3.0, 0.3, 20, 4
    logits = np.stack([zipf_logits(ctx, V_SMALL, 1.3) for _ in range(batch)])
    while True:  # draw uniforms until the float64 trace stays clear of every boundary
        uniforms = ctx.rng.random((steps, batch), dtype=np.float32)
        mu, clear = np.full(batch, 2 * tau), True
        for t in range(steps):
            _, mu, ok = model(logits, f32(mu), tau, eta, uniforms[t])
            clear &= bool(ok.all())
        if clear:
            break
    mu = np.full(batch, 2 * tau, dtype=np.float32)
    for t in range(steps):
        _, mu = check_step(ctx, logits, mu, tau, eta, uniforms[t])


def long_run(ctx, tau, eta, *, vocab=V_LARGE, batch=16, steps=250, burn_in=50, band=0.1):
    """Run `batch` sequences on one stationary distribution; return the mean surprise."""
    row = zipf_logits(ctx, vocab)
    logits = np.tile(row, (batch, 1))
    s = surprise_bits(row)
    top = int(row.argmax())
    mu = np.full(batch, 2 * tau, dtype=np.float32)
    uniforms = ctx.rng.random((steps, batch), dtype=np.float32)
    observed = np.empty((steps, batch))
    for t in range(steps):
        tokens, new_mu = call(ctx, logits, mu, tau, eta, uniforms[t])
        drawn = s[tokens]
        allowed = (drawn <= mu + 1e-3) | (tokens == top)
        check.that(
            bool(allowed.all()),
            f"token: step {t} drew a token whose surprise exceeds mu",
            f"mu = {mu}\ntokens = {tokens}\ntheir surprise in bits = {drawn}",
        )
        want = mu - eta * (drawn - tau)
        check.that(
            bool(np.abs(new_mu - want).max() <= 1e-3),
            f"new mu: step {t} is not mu - eta * (surprise of the drawn token - tau)",
            f"mu = {mu}\ntokens = {tokens}\nsurprise = {drawn}\nexpected {want}\ngot {new_mu}",
        )
        observed[t] = drawn
        mu = new_mu
    mean = float(observed[burn_in:].mean())
    ctx.metric("mean_surprise", round(mean, 3), "bits")
    check.that(
        abs(mean - tau) <= band,
        f"mean surprise: {mean:.3f} bits over steps {burn_in} to {steps - 1}, which is outside "
        f"the band of {band} bits around tau = {tau}",
        f"entropy of the distribution: {float((np.exp2(-s) * s).sum()):.3f} bits",
    )


@case(
    "250 steps on a stationary Zipf distribution: the mean surprise settles at tau = 3 bits",
    public=True,
)
def converges_to_tau(ctx):
    long_run(ctx, 3.0, 0.2)


@case("a tied maximum with nothing within mu: the lowest index of the tie is kept")
def tied_top(ctx):
    logits = f32(
        [
            [0.0, 2.0, 2.0, -1.0, 2.0, 0.5],
            [1.5, 1.5, 0.0, 0.0, -INF, 1.5],
            [-INF, -1.0, 3.0, 3.0, 0.0, 1.0],
        ]
    )
    tokens, _ = check_step(ctx, logits, f32([0.2, 0.4, 0.1]), 2.0, 0.1, f32([0.3, 0.6, 0.95]))
    check.equal(tokens, [1, 0, 2], label="token")


@case("masked tokens are never drawn, also with a large mu and u close to 1")
def masked_tokens(ctx):
    logits = ctx.randn(8, V_SMALL, scale=1.5)
    logits[:, [0, 5, V_SMALL - 1]] = -INF  # the last token is masked: an overshoot lands on it
    logits[3, :] = -INF
    logits[3, 7] = 0.25  # a single unmasked token: surprise 0
    mu = f32([50.0, 40.0, 3.0, 2.0, 100.0, 1.0, 60.0, 5.0])
    u = f32([0.99999, 0.0, 0.5, 0.9, 0.9999, 0.2, 0.999, 0.97])
    _, _, ok = model(logits, mu, 5.0, 0.1, u)
    u[~ok] = 0.5
    tokens, _ = check_step(ctx, logits, mu, 5.0, 0.1, u)
    check.that(
        bool(np.isfinite(logits[np.arange(8), tokens]).all()),
        "token: a masked token was drawn",
    )


@case("mu is the limit, not tau: rows with mu far above and far below the same tau")
def mu_not_tau(ctx):
    logits = np.tile(zipf_logits(ctx, V_SMALL, 1.0), (8, 1))
    mu = f32([0.6, 1.2, 2.2, 3.1, 4.3, 5.2, 6.4, 9.0])
    u = f32([0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93])
    while not model(logits, mu, 3.0, 0.1, u)[2].all():
        u = np.full(8, ctx.rng.uniform(0.85, 0.99), dtype=np.float32)
    check_step(ctx, logits, mu, 3.0, 0.1, u)


@case("a vocabulary of 200 tokens: random limits, then thirty steps fed back")
def large_vocabulary(ctx):
    tau, eta, steps, batch = 4.0, 0.15, 30, 4
    logits = np.stack([zipf_logits(ctx, V_LARGE) for _ in range(batch)])
    mu, u = draw_inputs(ctx, logits, 1.0, 9.0)
    check_step(ctx, logits, mu, tau, eta, u)
    while True:
        uniforms = ctx.rng.random((steps, batch), dtype=np.float32)
        mu, clear = np.full(batch, 2 * tau), True
        for t in range(steps):
            _, mu, ok = model(logits, f32(mu), tau, eta, uniforms[t])
            clear &= bool(ok.all())
        if clear:
            break
    mu = np.full(batch, 2 * tau, dtype=np.float32)
    for t in range(steps):
        _, mu = check_step(ctx, logits, mu, tau, eta, uniforms[t])


@case("250 steps with tau = 4.5 bits and a faster learning rate: the mean surprise settles")
def converges_to_higher_tau(ctx):
    long_run(ctx, 4.5, 0.4, band=0.15)


@case("a single row and a vocabulary of one token")
def single_row_single_token(ctx):
    logits = ctx.randn(1, V_SMALL, scale=2.0)
    mu, u = draw_inputs(ctx, logits, 1.0, 6.0)
    check_step(ctx, logits, mu, 3.0, 0.1, u)
    tokens, new_mu = check_step(ctx, f32([[7.5], [-3.0]]), f32([4.0, -1.0]), 2.0, 0.5, [0.0, 0.9])
    check.equal(tokens, [0, 0], label="token")
    check.allclose(new_mu, f32([5.0, 0.0]), atol=1e-5, label="new mu")
