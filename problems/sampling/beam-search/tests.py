"""Tests for beam-search.

Every case checks the hypotheses first, then the cost.

The expected hypotheses come from `definition` below: beam search exactly as the
statement defines it, written without a cache, without batching, and in float64 (one
full-prefix model call per live beam and step). It shares no code with the reference.
It also counts the steps the search takes, which gives the cost limits, and tracks the
smallest gap in any comparison it makes (between neighboring candidates, between
finished scores, between the stopping bound and the worst finished score). Cached and
uncached logits agree to about 1e-5, so every fixture must keep that gap above
MARGIN; a fixture that does not is an authoring error.
"""

from itertools import pairwise

import numpy as np

from learnai.judge import case, check
from learnai.kit import TinyLM

FN = "beam_search"
MARGIN = 1e-3


def log_softmax64(logits):
    x = np.asarray(logits, dtype=np.float64)
    top = x.max()
    return x - top - np.log(np.exp(x - top).sum())


def definition(ctx, seed, prompt, n_beams, max_new_tokens, eos_id, length_penalty):
    """Returns (hypotheses best first, number of steps, smallest comparison gap)."""
    fw = ctx.oracle_fw
    lm = TinyLM(fw, seed=seed)
    beams, finished, margin = [([], 0.0)], [], np.inf
    for step in range(1, max_new_tokens + 1):
        candidates = []
        for row, (tokens, total) in enumerate(beams):
            log_probs = log_softmax64(fw.numpy(lm([*prompt, *tokens])[0])[-1])
            candidates += [(total + lp, row, v, tokens) for v, lp in enumerate(log_probs)]
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        top = candidates[: 2 * n_beams + 1]
        margin = min(margin, *(a[0] - b[0] for a, b in pairwise(top)))
        live = []
        for rank, (total, _, token, tokens) in enumerate(top[: 2 * n_beams]):
            if step == max_new_tokens or token == eos_id:
                if rank < n_beams:
                    finished.append((total / step**length_penalty, [*tokens, token]))
            elif len(live) < n_beams:
                live.append(([*tokens, token], total))
        finished.sort(key=lambda hypothesis: -hypothesis[0])
        for a, b in pairwise(finished):
            margin = min(margin, a[0] - b[0])
        del finished[n_beams:]
        if step == max_new_tokens:
            break
        if len(finished) == n_beams:
            bound = live[0][1] / max_new_tokens**length_penalty
            margin = min(margin, abs(bound - finished[-1][0]))
            if bound <= finished[-1][0]:
                break
        beams = live
    return [(tokens, score) for score, tokens in finished], step, margin


def search(ctx, seed, prompt, n_beams, max_new_tokens, eos_id, length_penalty, steps=None):
    """Run the solution; check its hypotheses, then its cost. Returns the hypotheses."""
    expected, n_steps, margin = definition(
        ctx, seed, prompt, n_beams, max_new_tokens, eos_id, length_penalty
    )
    if margin < MARGIN or (steps is not None and steps != n_steps):
        raise RuntimeError(f"authoring error: margin {margin:.1e}, {n_steps} steps")
    lm = TinyLM(ctx.fw, seed=seed)
    got = ctx.call(FN, lm, list(prompt), n_beams, max_new_tokens, eos_id, length_penalty)
    check.that(
        isinstance(got, list)
        and all(
            isinstance(h, tuple)
            and len(h) == 2
            and isinstance(h[0], list)
            and all(type(t) is int for t in h[0])
            and isinstance(h[1], float)
            for h in got
        ),
        f"{FN}: must return a list of (tokens, score) tuples: a list of Python ints and a float",
        f"got {got!r}",
    )
    show = "\n".join(f"  {tokens}  {score:.4f}" for tokens, score in expected)
    detail = f"expected, best first:\n{show}\ngot:\n" + "\n".join(
        f"  {tokens}  {score:.4f}" for tokens, score in got
    )
    check.that(
        len(got) == n_beams,
        f"{FN}: must return exactly n_beams hypotheses",
        f"n_beams = {n_beams}, got {len(got)} hypotheses\n{detail}",
    )
    check.that(
        [tokens for tokens, _ in got] == [tokens for tokens, _ in expected],
        f"{FN}: wrong hypotheses (their tokens, or their order)",
        detail,
    )
    check.allclose(
        np.array([score for _, score in got]),
        np.array([score for _, score in expected]),
        rtol=1e-4,
        atol=1e-5,
        label="scores",
    )
    check.cost(lm.counters, "forward_calls", at_most=n_steps)
    check.cost(
        lm.counters,
        "tokens_processed",
        at_most=len(prompt) + n_beams * (n_steps - 1),
        unit="tokens",
    )
    return got


@case("the worked example: two beams, the search stops after 3 of 12 steps", public=True)
def worked_example(ctx):
    got = search(ctx, 6, [0, 63, 33, 13], 2, 12, 52, 0.0, steps=3)
    check.equal([tokens for tokens, _ in got], [[52], [38, 4, 52]], label="hypotheses")
    check.allclose(np.array([s for _, s in got]), [-1.3500, -3.6046], atol=1e-3, label="scores")


@case("length penalty 1.0: a longer hypothesis found at the last step displaces one", public=True)
def length_penalty_example(ctx):
    # Catches stopping when n_beams hypotheses have finished, and bounding the live
    # beams with their current length: both stop at step 3 and return [5, 5, 36].
    got = search(ctx, 10, [0, 43, 22, 24, 46], 2, 5, 36, 1.0, steps=5)
    check.equal(
        [tokens for tokens, _ in got], [[5, 36], [35, 61, 39, 29, 58]], label="hypotheses"
    )


@case("three beams, EOS never generated: every hypothesis has max_new_tokens tokens", public=True)
def no_eos(ctx):
    # The beams are reshuffled at almost every step: catches a cache whose rows are
    # not reordered to follow them.
    search(ctx, 10, [0, 18, 53], 3, 7, 60, 1.0, steps=7)


@case("five beams, no length penalty: the search stops after 6 of 13 steps", public=True)
def five_beams_early_stop(ctx):
    search(ctx, 8, [0, 54, 51, 62], 5, 13, 38, 0.0, steps=6)


@case("early stop with a length penalty: 13 of 14 steps, 10 of 14 steps")
def early_stop_with_penalty(ctx):
    search(ctx, 8, [0, 17, 28], 4, 14, 38, 1.0, steps=13)
    search(ctx, 10, [0, 51, 34, 58, 11], 2, 14, 35, 0.5, steps=10)


@case("EOS candidates ranked below n_beams are dropped, not finished")
def low_ranked_eos(ctx):
    search(ctx, 6, [0, 26, 18, 16, 54], 2, 10, 52, 0.0, steps=4)
    search(ctx, 9, [0, 45], 4, 14, 62, 0.5, steps=14)


@case("several beams finish in one step and the beam stays n_beams wide")
def stays_wide(ctx):
    # Catches taking only n_beams candidates, which loses a live beam per EOS.
    search(ctx, 6, [0, 38, 11, 2], 5, 12, 52, 0.0, steps=12)
    search(ctx, 2, [0, 29], 2, 11, 22, 0.0, steps=5)


@case("max_new_tokens = 1: the prefill is the only call")
def single_step(ctx):
    search(ctx, 3, [0, 17, 42, 5], 3, 1, 1, 1.0, steps=1)
    search(ctx, 3, [0, 17, 42, 5], 2, 1, None, 0.0, steps=1)


@case("n_beams = 1, and eos_id = None")
def one_beam_and_no_eos(ctx):
    search(ctx, 3, [0, 62, 28], 1, 8, 40, 0.0, steps=2)
    search(ctx, 7, [0, 53, 15, 5], 1, 8, 1, 1.0, steps=8)
    search(ctx, 2, [0, 6, 17], 4, 6, None, 0.0, steps=6)


@case("a single-token prompt with length penalties 2.0 and 0.5")
def single_token_prompt(ctx):
    search(ctx, 5, [0], 2, 7, 58, 2.0, steps=7)
    search(ctx, 5, [0], 3, 9, 37, 0.5, steps=9)


@case("a 30-token prompt: it is processed once, in the first call")
def long_prompt(ctx):
    # The length penalty counts generated tokens only, never the prompt.
    search(ctx, 7, [0, *range(20, 49)], 3, 6, 18, 1.0, steps=6)
    search(ctx, 7, [0, *range(20, 49)], 3, 6, 58, 0.0, steps=6)
