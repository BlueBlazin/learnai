"""Tests for reasoning-budget-processor.

`drive` plays a generation: at every step it hands the same raw logits and the same
previous token to the solution and to the oracle (each with its own state), compares
the processed logits, and then chooses the next token itself from the oracle's row: a
scripted token where the case wants one (the model opening or closing the span), the
argmax otherwise. Because the test chooses, a wrong row cannot change what the
solution is asked next, and the first wrong step is the one reported.

The cost cases let the solution run a `TinyLM` on its own: the token is the argmax of
the solution's row, and the test counts the thinking tokens the solution let through.
"""

import numpy as np

from learnai.judge import CheckFailure, case, check
from learnai.kit import TinyLM

FN = "budget_step"
V = 8
START, END = 6, 7


def config(budget, *, forced=(5, END), allowed=(0, 1, 2), start=START, end=END, size=V):
    mask = None
    if allowed is not None:
        mask = np.zeros(size, dtype=bool)
        mask[list(allowed)] = True
    return {
        "start_id": start,
        "end_id": end,
        "forced": list(forced),
        "budget": budget,
        "answer_mask": mask,
    }


def describe(cfg):
    mask = cfg["answer_mask"]
    allowed = None if mask is None else np.flatnonzero(mask).tolist()
    return (
        f"cfg: start_id={cfg['start_id']}, end_id={cfg['end_id']}, forced={cfg['forced']}, "
        f"budget={cfg['budget']}, answer tokens={allowed}"
    )


def compare(got, want, step, history, cfg):
    """Compare one processed row; the message says what kind of difference it is."""
    where = f"call {step + 1}, after the tokens {history}\n{describe(cfg)}\n"
    got = np.asarray(got)
    check.shape(got, want.shape, label=f"{FN}: logits")
    check.that(
        got.dtype == np.float32,
        f"{FN}: the returned logits have dtype {got.dtype}, expected dtype float32 (the dtype "
        "of the input)",
    )
    blocked, open_ = np.isneginf(want), np.isfinite(want)
    leaked = np.flatnonzero(blocked & ~np.isneginf(got))
    lost = np.flatnonzero(open_ & ~np.isfinite(got))
    detail = f"{where}expected = {want}\ngot      = {got}"
    if leaked.size:
        raise CheckFailure(
            f"{FN}: call {step + 1} leaves token {int(leaked[0])} available, "
            "which may not come next at this point",
            detail,
        )
    if lost.size:
        raise CheckFailure(
            f"{FN}: call {step + 1} blocks token {int(lost[0])}, "
            "which must stay available at this point",
            detail,
        )
    try:
        check.allclose(got, want, label=f"{FN}: logits of call {step + 1}")
    except CheckFailure as failure:
        raise CheckFailure(
            f"{FN}: call {step + 1} changed the logit of a token that stays available",
            detail,
        ) from failure


def drive(ctx, cfg, rows, script=None, pick=None):
    """Run len(rows) steps. Returns the chosen tokens and the oracle's rows."""
    script = script or {}
    state_s = state_o = token = None
    tokens, wants = [], []
    for step, row in enumerate(rows):
        state_o, want = ctx.expected(FN, state_o, token, row, cfg)
        result = ctx.call(FN, state_s, token, row, cfg)
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            f"{FN}: must return a tuple (state, logits)",
            f"got {type(result).__name__}",
        )
        state_s, got = result
        want = np.asarray(want)
        compare(got, want, step, tokens, cfg)
        if step in script:
            token = script[step]
            if not np.isfinite(want[token]):
                raise RuntimeError(f"authoring error: token {token} is blocked at step {step}")
        elif pick is not None:
            token = int(pick(np.flatnonzero(np.isfinite(want))))
        else:
            token = int(want.argmax())
        tokens.append(token)
        wants.append(want)
    return tokens, wants


def random_rows(ctx, n, size=V):
    """Raw logits whose argmax is never the start or end token: those are scripted."""
    rows = ctx.randn(n, size, scale=1.5)
    rows[:, [START, END]] -= 8.0
    return rows


# The worked example of the statement.
EXAMPLE_ROWS = np.array(
    [
        [0.5, 0.1, 0.3, 2.0, 0.2, 0.0, 1.5, 0.4],
        [0.1, 0.2, 0.0, 0.3, 1.2, 0.5, 0.4, 0.6],
        [0.3, 0.1, 0.2, 1.4, 0.5, 0.0, 0.6, 0.9],
        [1.0, 0.2, 0.1, 0.4, 0.8, -0.5, 0.3, 0.6],
        [0.2, 0.9, 0.1, 0.3, 0.4, 0.5, 0.6, 0.0],
        [0.1, 0.4, 0.7, 1.1, 0.9, 0.2, 1.3, 0.5],
        [0.8, 0.3, 0.2, 0.1, 0.6, 0.4, 0.0, 1.0],
    ],
    dtype=np.float32,
)
EXAMPLE_OPEN = [(0, 1, 2, 6), None, None, (5,), (7,), (0, 1, 2), (0, 1, 2)]  # None: all
EXAMPLE_TOKENS = [6, 4, 3, 5, 7, 2, 0]


@case("the worked example: budget 2, the end sequence [5, 7], an answer mask", public=True)
def worked_example(ctx):
    cfg = config(2)
    tokens, wants = drive(ctx, cfg, EXAMPLE_ROWS)
    for row, want, open_ in zip(EXAMPLE_ROWS, wants, EXAMPLE_OPEN, strict=True):
        literal = row.copy()
        if open_ is not None:
            literal[[i for i in range(V) if i not in open_]] = -np.inf
        if not np.array_equal(literal, want):
            raise RuntimeError("authoring error: the oracle disagrees with the statement")
    if tokens != EXAMPLE_TOKENS:
        raise RuntimeError(f"authoring error: the example's tokens are {tokens}")


@case("the model closes the span itself after 1 of 5 tokens: nothing is forced", public=True)
def closes_early(ctx):
    # Token 5, the first token of the end sequence, later appears in the answer of the
    # unconstrained run and means nothing there.
    rows = random_rows(ctx, 14)
    drive(ctx, config(5), rows, script={0: START, 2: END})
    drive(ctx, config(5, allowed=None), rows, script={0: START, 2: END, 4: 5, 5: END})


@case("a budget of 0: the end sequence is forced right after the start token", public=True)
def zero_budget(ctx):
    rows = random_rows(ctx, 8)
    tokens, _ = drive(ctx, config(0), rows, script={0: START})
    if tokens[:3] != [START, 5, END]:
        raise RuntimeError("authoring error: expected the end sequence after the start token")
    # The same budget when the model answers directly: nothing to close.
    drive(ctx, config(0), rows, script={0: 1})


@case("a three-token end sequence is forced one token per step, to its end", public=True)
def long_end_sequence(ctx):
    rows = random_rows(ctx, 12)
    for allowed in ((0, 1, 2), None):
        cfg = config(3, forced=(4, 5, END), allowed=allowed)
        tokens, _ = drive(ctx, cfg, rows, script={0: START})
        if tokens[4:7] != [4, 5, END]:
            raise RuntimeError("authoring error: expected the end sequence at steps 4 to 6")


def model_run(ctx, seed, cfg, steps, close_at=None):
    """The solution drives a TinyLM. Returns the number of thinking tokens it let through.

    The raw logits get +50 on the start token at step 0 (the model opens a span), and
    on the end token -50 (the model does not close the span) except at step `close_at`,
    where it gets +50.
    """
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    start, end, forced = cfg["start_id"], cfg["end_id"], cfg["forced"]
    logits, cache = lm([0, 17, 42, 5])
    state = token = None
    tokens, free = [], []
    for step in range(steps):
        row = ctx.oracle_fw.numpy(logits)[-1].copy()
        if step == 0:
            row[start] += 50.0
        row[end] += 50.0 if step == close_at else -50.0
        state, got = ctx.call(FN, state, token, row, cfg)
        got = np.asarray(got)
        check.that(
            bool(np.isfinite(got).any()) and not np.isnan(got).any(),
            f"{FN}: call {step + 1} returned a row with no token left to choose (or NaN)",
        )
        token = int(got.argmax())
        tokens.append(token)
        free.append(int(np.isfinite(got).sum()) > 1)
        logits, cache = lm([token], cache)
    history = f"generated tokens: {tokens}\n{describe(cfg)}"
    check.that(tokens[0] == start, f"{FN}: the start-of-thinking token was blocked", history)
    check.that(
        end in tokens,
        f"{FN}: the thinking span was never closed in {steps} generated tokens",
        history,
    )
    close = tokens.index(end)
    thinking = sum(free[1:close])
    check.at_most("thinking_tokens", thinking, cfg["budget"], unit="tokens")
    if close_at is None:
        check.that(
            tokens[close - len(forced) + 1 : close + 1] == forced and close >= len(forced),
            f"{FN}: the span was not closed by the complete end sequence",
            history,
        )
        check.that(
            thinking == cfg["budget"],
            f"{FN}: the end sequence was forced after {thinking} thinking tokens, "
            "before the budget was used up",
            history,
        )
    mask = cfg["answer_mask"]
    if mask is not None:
        outside = [t for t in tokens[close + 1 :] if not mask[t]]
        check.that(
            not outside,
            f"{FN}: the answer contains token {outside[0] if outside else 0}, "
            "which the answer mask does not allow",
            history,
        )
    return thinking


def lm_config(budget, forced, allowed):
    return config(budget, forced=forced, allowed=allowed, start=2, end=3, size=64)


@case("a TinyLM that thinks for as long as it may: at most `budget` thinking tokens", public=True)
def model_run_budget(ctx):
    # This model never produces the end token on its own.
    digits = range(10, 20)
    model_run(ctx, 3, lm_config(8, (40, 41, 3), digits), 24)
    model_run(ctx, 3, lm_config(0, (3,), digits), 6)


@case("no span: the first token is an answer token, and a later start token opens nothing")
def no_span(ctx):
    rows = random_rows(ctx, 16)
    # Constrained: the first row is masked to the answer tokens and the start token.
    drive(ctx, config(3), rows, script={0: 2})
    # Unconstrained: the start token appears at step 3, inside the answer. No span
    # opens, so nothing may be forced for the rest of the run.
    drive(ctx, config(3, allowed=None), rows, script={0: 2, 3: START})
    drive(ctx, config(0, allowed=None), rows, script={0: 2, 3: START})


@case("the span is closed by the model at the last token the budget allows")
def closes_at_the_limit(ctx):
    rows = random_rows(ctx, 12)
    # Thinking tokens at steps 1 to 3, the end token at step 4: 3 of 4 used.
    drive(ctx, config(4), rows, script={0: START, 4: END})
    # A budget of 1: one thinking token, then the forced sequence.
    tokens, _ = drive(ctx, config(1), rows, script={0: START})
    if tokens[2:4] != [5, END]:
        raise RuntimeError("authoring error: expected the end sequence at steps 2 and 3")
    # The start token inside the span is an ordinary thinking token.
    drive(ctx, config(3, allowed=None), rows, script={0: START, 1: START, 2: START})


@case("a budget larger than the run, and a one-token end sequence")
def large_budget_single_end(ctx):
    rows = random_rows(ctx, 20)
    drive(ctx, config(1000), rows, script={0: START})
    tokens, _ = drive(ctx, config(4, forced=(END,)), rows, script={0: START})
    if tokens[5] != END:
        raise RuntimeError("authoring error: expected the end token at step 5")


@case("a forced token with a very low logit keeps its logit")
def forced_keeps_logit(ctx):
    rows = random_rows(ctx, 10)
    rows[:, 5] = -30.0
    rows[:, END] = -45.5
    drive(ctx, config(2), rows, script={0: START})


@case("random configurations, with the next token drawn among the available ones")
def random_runs(ctx):
    for trial in range(8):
        rng = ctx.child_rng(trial)
        ids = rng.permutation(V).tolist()
        start, end = ids[0], ids[1]
        message = ids[2 : 2 + int(rng.integers(0, 3))]
        allowed = None if trial % 3 == 0 else sorted(ids[4 : 4 + int(rng.integers(1, 4))])
        cfg = config(
            int(rng.integers(0, 6)), forced=(*message, end), allowed=allowed, start=start, end=end
        )
        rows = rng.normal(0.0, 1.5, (18, V)).astype(np.float32)
        script = {0: start} if trial % 4 else {}
        drive(ctx, cfg, rows, script=script, pick=rng.choice)


@case("TinyLM runs on another model: an early close, a zero budget, no answer mask")
def model_runs_hidden(ctx):
    letters = range(20, 40)
    model_run(ctx, 7, lm_config(6, (50, 3), letters), 20)
    model_run(ctx, 7, lm_config(6, (50, 3), letters), 20, close_at=3)
    model_run(ctx, 11, lm_config(12, (3,), None), 30)
    model_run(ctx, 11, lm_config(0, (44, 45, 46, 3), letters), 12)
