"""Tests for diffusion-unmask-step.

Numeric cases compare `unmask_step` with the oracle exactly. Two float32 decisions
could differ between frameworks in the last bits, so `plan` recomputes every case in
float64 and refuses inputs that are too close to call (an authoring error, not a
solution failure): a uniform number within SAMPLE_MARGIN of a cumulative probability
at a masked position, the two best logits of a masked position within LOGIT_MARGIN
at temperature 0, and the confidences on either side of the cut within RANK_MARGIN,
unless the case is about ties and builds them from identical rows. `draw` redraws
uniforms until the first condition holds.

The statistical cases use one call with N masked positions that share one row of
logits and commit all of them: the committed tokens are then N independent draws from
softmax(row / temperature) without the mask token. Three checks at the default delta
of 1e-10 keep the false-failure rate of a correct solution below 1e-9.
"""

import numpy as np

from learnai.judge import case, check

STEP = "unmask_step"
SAMPLE_MARGIN = 1e-4
LOGIT_MARGIN = 1e-3
RANK_MARGIN = 1e-4
N = 20_000


def f32(values):
    return np.array(values, dtype=np.float32)


def softmax64(logits, mask_id):
    x = np.asarray(logits, dtype=np.float64).copy()
    x[:, mask_id] = -np.inf
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def plan(tokens, logits, mask_id, n_unmask, temperature, uniforms):
    """The statement in float64: (new tokens, sampling gap, logit gap, rank gap)."""
    tokens = np.asarray(tokens)
    masked = tokens == mask_id
    probs = softmax64(logits, mask_id)
    sample_gap = logit_gap = np.inf
    if temperature == 0:
        candidates = probs.argmax(axis=-1)
        top = np.sort(np.delete(np.asarray(logits, dtype=np.float64), mask_id, axis=1), axis=-1)
        if masked.any() and top.shape[1] > 1:
            logit_gap = float((top[masked, -1] - top[masked, -2]).min())
    else:
        cdf = np.cumsum(softmax64(np.asarray(logits, dtype=np.float64) / temperature, mask_id), -1)
        threshold = np.asarray(uniforms, dtype=np.float64)[:, None] * cdf[:, -1:]
        candidates = (cdf <= threshold).sum(axis=-1)
        if masked.any():
            sample_gap = float(np.abs(cdf - threshold)[masked].min())
    confidence = np.take_along_axis(probs, candidates[:, None], axis=-1)[:, 0]
    confidence = np.where(masked, confidence, -np.inf)
    n = min(n_unmask, int(masked.sum()))
    order = np.argsort(-confidence, kind="stable")
    rank_gap = np.inf
    if 0 < n < int(masked.sum()):
        rank_gap = float(confidence[order[n - 1]] - confidence[order[n]])
    new = tokens.copy()
    new[order[:n]] = candidates[order[:n]]
    return new, sample_gap, logit_gap, rank_gap


def draw(ctx, tokens, logits, mask_id, temperature):
    """Uniforms in [0, 1), one per position, with no masked position near a boundary."""
    u = ctx.rng.random(len(tokens), dtype=np.float32)
    for _ in range(100):
        gap = plan(tokens, logits, mask_id, 0, temperature, u)[1]
        if gap >= SAMPLE_MARGIN:
            return u
        u = ctx.rng.random(len(tokens), dtype=np.float32)
    raise RuntimeError("authoring error: could not draw unambiguous uniforms")


def check_step(ctx, tokens, logits, mask_id, n_unmask, temperature, uniforms, *, ties=False):
    """Call the solution, check the form of its result, and compare with the oracle."""
    tokens, logits, uniforms = np.asarray(tokens), f32(logits), f32(uniforms)
    want, sample_gap, logit_gap, rank_gap = plan(
        tokens, logits, mask_id, n_unmask, temperature, uniforms
    )
    if sample_gap < SAMPLE_MARGIN or logit_gap < LOGIT_MARGIN:
        raise RuntimeError("authoring error: a candidate is too close to call")
    if not ties and rank_gap < RANK_MARGIN:
        raise RuntimeError("authoring error: two confidences at the cut are too close")
    got = np.asarray(ctx.call(STEP, tokens, logits, mask_id, n_unmask, temperature, uniforms))
    detail = (
        f"tokens = {tokens.tolist()}  (mask_id = {mask_id})\nn_unmask = {n_unmask}, "
        f"temperature = {temperature}\nresult = {got.tolist()}\nexpected = {want.tolist()}"
    )
    check.shape(got, tokens.shape, label=STEP)
    check.that(
        got.dtype.kind in "iu",
        f"{STEP}: expected an integer tensor of tokens, got dtype {got.dtype}",
    )
    masked = tokens == mask_id
    changed = np.flatnonzero(~masked & (got != tokens))
    check.that(
        changed.size == 0,
        f"{STEP}: position {int(changed[0]) if changed.size else 0} was not masked and its "
        "token was changed",
        detail,
    )
    expected = ctx.expected(STEP, tokens, logits, mask_id, n_unmask, temperature, uniforms)
    if not np.array_equal(np.asarray(expected), want):
        raise RuntimeError("authoring error: the oracle disagrees with the float64 plan")
    committed = int((masked & (got != mask_id)).sum())
    due = int((masked & (want != mask_id)).sum())
    check.that(
        committed == due,
        f"{STEP}: {committed} masked positions were committed, expected {due} (the number "
        "this step must commit)",
        detail,
    )
    check.equal(got, want, label="tokens after the step")
    return got


# The worked example of the statement: V = 5, the mask token is 4.
M = 4
EXAMPLE_TOKENS = [3, M, M, 1, M, M]
EXAMPLE_PROBS = [
    [0.90, 0.04, 0.03, 0.03],
    [0.70, 0.10, 0.10, 0.10],
    [0.25, 0.25, 0.40, 0.10],
    [0.25, 0.25, 0.25, 0.25],
    [0.10, 0.60, 0.20, 0.10],
    [0.50, 0.30, 0.10, 0.10],
]
EXAMPLE_UNIFORMS = [0.2, 0.5, 0.95, 0.4, 0.3, 0.6]


def example_logits():
    # log of the probabilities, and a logit of 1.0 for the mask token in every row.
    logits = np.log(np.array(EXAMPLE_PROBS, dtype=np.float64))
    return f32(np.concatenate([logits, np.ones((6, 1))], axis=1))


@case("the worked example: three of four masked positions are committed", public=True)
def worked_example(ctx):
    got = check_step(ctx, EXAMPLE_TOKENS, example_logits(), M, 3, 1.0, EXAMPLE_UNIFORMS)
    check.equal(got, [3, 0, M, 1, 1, 1], label="tokens after the step")


@case("schedule: 10 masks in 4 steps, 3 in 5, 12 in 4, none", public=True)
def schedule_examples(ctx):
    for n_masked, steps, want in [
        (10, 4, [3, 3, 2, 2]),
        (3, 5, [1, 1, 1, 0, 0]),
        (12, 4, [3, 3, 3, 3]),
        (0, 3, [0, 0, 0]),
        (7, 1, [7]),
    ]:
        got = ctx.call("schedule", n_masked, steps)
        check.that(
            isinstance(got, list) and all(type(n) is int for n in got),
            "schedule: must return a list of Python ints",
            f"schedule({n_masked}, {steps}) returned {got!r}",
        )
        check.equal(got, want, label=f"schedule({n_masked}, {steps})")


@case("temperature 0: every candidate is the most probable token", public=True)
def greedy_candidates(ctx):
    logits = ctx.randn(12, 16, scale=2.0)
    tokens = ctx.randint(0, 15, 12)
    tokens[[0, 2, 3, 6, 7, 9, 11]] = 15
    uniforms = ctx.rng.random(12, dtype=np.float32)
    for n_unmask in (1, 3, 7):
        check_step(ctx, tokens, logits, 15, n_unmask, 0.0, uniforms)


@case("the committed tokens follow softmax(logits): 20,000 masked positions", public=True)
def follows_distribution(ctx):
    follows(ctx, ctx.randn(8, scale=1.2), 1.0, mask_id=7)


def follows(ctx, row, temperature, mask_id):
    """All N positions are masked and share `row`; every one of them is committed."""
    logits = np.tile(f32(row), (N, 1))
    tokens = np.full(N, mask_id)
    uniforms = ctx.rng.random(N, dtype=np.float32)
    got = np.asarray(ctx.call(STEP, tokens, logits, mask_id, N, temperature, uniforms))
    check.shape(got, (N,), label=STEP)
    target = softmax64(np.asarray(row, dtype=np.float64)[None] / temperature, mask_id)[0]
    check.distribution(got, target, label="committed token")


def reverse_process(ctx, length, vocab, steps, temperature, prompt=0):
    """A whole generation: `steps` steps with the solution's schedule, each step checked.

    The "model" is a table of logits per step, so the oracle and the solution see the
    same predictions whatever they committed before.
    """
    mask_id = vocab - 1
    tokens = np.full(length, mask_id)
    tokens[:prompt] = ctx.randint(0, mask_id, prompt)
    counts = ctx.call("schedule", length - prompt, steps)
    check.equal(counts, ctx.expected("schedule", length - prompt, steps), label="schedule")
    for n_unmask in counts:
        logits = ctx.randn(length, vocab, scale=2.0)
        uniforms = draw(ctx, tokens, logits, mask_id, temperature)
        tokens = check_step(ctx, tokens, logits, mask_id, n_unmask, temperature, uniforms)
    left = int((tokens == mask_id).sum())
    check.that(left == 0, f"{left} positions are still masked after the last step")


@case("a whole reverse process: 12 positions in 5 steps, no mask left at the end", public=True)
def whole_process(ctx):
    reverse_process(ctx, 12, 16, 5, 0.7)


@case("the drawn token is not the favorite: confidence is that of the drawn token")
def confidence_of_the_draw(ctx):
    # Rows 0 and 1 have the same favorite (probability 0.6), but row 0 draws its
    # 0.15 token. Ranked by the favorite they tie; ranked correctly row 1 comes first.
    probs = [[0.6, 0.15, 0.15, 0.1], [0.6, 0.15, 0.15, 0.1], [0.3, 0.3, 0.2, 0.2]]
    probs += [[0.05, 0.05, 0.85, 0.05], [0.45, 0.35, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1]]
    logits = f32(np.concatenate([np.log(probs), np.zeros((6, 1))], axis=1))
    uniforms = [0.7, 0.3, 0.1, 0.5, 0.6, 0.95]
    for n_unmask in (1, 2, 3, 4):
        check_step(ctx, [M, M, M, M, M, M], logits, M, n_unmask, 1.0, uniforms)


@case("unmasked positions with confident predictions are never touched")
def unmasked_stay(ctx):
    logits = ctx.randn(12, 16, scale=1.0)
    tokens = ctx.randint(0, 15, 12)
    masked = [1, 4, 5, 8, 10]
    tokens[masked] = 15
    # The model is nearly certain at the unmasked positions, of a different token.
    for i in range(12):
        if i not in masked:
            logits[i, (tokens[i] + 1) % 15] += 12.0
    for temperature in (0.0, 1.0):
        uniforms = draw(ctx, tokens, logits, 15, temperature)
        for n_unmask in (2, 5):
            check_step(ctx, tokens, logits, 15, n_unmask, temperature, uniforms)


@case("n_unmask larger than the number of masks, and n_unmask = 0")
def more_than_masked(ctx):
    logits = ctx.randn(12, 16, scale=2.0)
    tokens = ctx.randint(0, 15, 12)
    tokens[[3, 7]] = 15
    uniforms = draw(ctx, tokens, logits, 15, 1.0)
    for n_unmask in (0, 2, 5, 12, 40):
        check_step(ctx, tokens, logits, 15, n_unmask, 1.0, uniforms)
    # Nothing is masked: nothing changes.
    done = ctx.randint(0, 15, 12)
    check_step(ctx, done, logits, 15, 3, 1.0, uniforms)


@case("the mask token has the largest logit everywhere and is still never predicted")
def mask_token_excluded(ctx):
    for mask_id in (0, 9, 15):
        logits = ctx.randn(12, 16, scale=1.5)
        logits[:, mask_id] = 9.0
        tokens = np.full(12, mask_id)
        tokens[[0, 5]] = (mask_id + 1) % 16
        for temperature in (0.0, 1.0):
            uniforms = draw(ctx, tokens, logits, mask_id, temperature)
            got = check_step(ctx, tokens, logits, mask_id, 10, temperature, uniforms)
            check.that(not (got == mask_id).any(), "a position was committed to the mask token")


@case("equal confidences: the lowest index is committed first")
def ties(ctx):
    # Identical rows give bit-identical confidences at temperature 0.
    row = ctx.randn(16, scale=2.0)
    logits = np.tile(row, (12, 1))
    logits[4] = ctx.randn(16, scale=0.2)  # one position the model is unsure about
    tokens = np.full(12, 15)
    tokens[[0, 6]] = 3
    uniforms = ctx.rng.random(12, dtype=np.float32)
    for n_unmask in (1, 2, 4, 9):
        check_step(ctx, tokens, logits, 15, n_unmask, 0.0, uniforms, ties=True)


@case("each position is drawn with its own uniform number, at two temperatures")
def own_uniforms(ctx):
    logits = ctx.randn(12, 16, scale=1.5)
    tokens = ctx.randint(0, 15, 12)
    tokens[[2, 3, 5, 8, 9, 11]] = 15
    for temperature in (0.5, 1.0, 2.0):
        for _ in range(3):
            uniforms = draw(ctx, tokens, logits, 15, temperature)
            for n_unmask in (2, 6):
                check_step(ctx, tokens, logits, 15, n_unmask, temperature, uniforms)


@case("temperature 0.5 sharpens the committed tokens: 20,000 masked positions")
def follows_tempered(ctx):
    follows(ctx, ctx.randn(8, scale=1.0), 0.5, mask_id=0)


@case("temperature 2 flattens them, with a rare token: 20,000 masked positions")
def follows_flattened(ctx):
    with np.errstate(divide="ignore"):
        row = np.log(f32([0.5, 0.2, 0.004, 0.1, 0.05, 0.1, 0.03, 0.016]))
    row[3] = 6.0  # the mask token, with the largest logit
    follows(ctx, row, 2.0, mask_id=3)


@case("schedules: the counts add up for every (n_masked, steps) up to 40 by 12")
def schedule_sweep(ctx):
    for n_masked in range(41):
        for steps in range(1, 13):
            got = ctx.call("schedule", n_masked, steps)
            want = ctx.expected("schedule", n_masked, steps)
            check.that(
                isinstance(got, list) and len(got) == steps,
                "schedule: the list does not have one entry per step",
                f"schedule({n_masked}, {steps}) returned {got!r}",
            )
            check.that(
                sum(got) == n_masked,
                f"schedule({n_masked}, {steps}) commits {sum(got)} positions in total, "
                f"expected {n_masked} (the number of masked positions)",
                f"returned {got!r}",
            )
            check.equal(got, want, label=f"schedule({n_masked}, {steps})")


@case("whole reverse processes: a prompt, more steps than masks, one step")
def more_processes(ctx):
    reverse_process(ctx, 12, 16, 4, 1.0, prompt=3)
    reverse_process(ctx, 12, 16, 12, 0.0, prompt=7)
    reverse_process(ctx, 12, 16, 1, 1.3)
