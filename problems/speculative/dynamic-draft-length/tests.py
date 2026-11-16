"""Tests for dynamic-draft-length.

Everything is exact: costs are integers and the policy is fully specified, so
`simulate` is compared with the oracle key by key. Confidences are compared with a
threshold, so every trace is built from target confidences that stay at least 1e-6
away from the thresholds the cases use (`logits_with_confidence` asserts it); the one
exception is the strictness case, whose confidence is exactly 0.5.

The cost cases bound the adaptive policy's cost by 0.9 times the best constant draft
length, which the judge computes with the oracle.
"""

import math

import numpy as np

from learnai.judge import case, check

T, F = True, False
THRESHOLDS = (0.0, 0.25, 0.4, 0.5, 0.7)

# The worked example of the statement.
WORKED = [
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [2.0, 0.0]),
    (F, [0.2, 0.0]),
    (T, [3.0, 0.0]),
    (F, [0.0, 0.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
    (T, [3.0, 0.0]),
]


def logits_with_confidence(rng, confidence, vocab=4):
    """Logits whose largest softmax probability is `confidence`, at a random index."""
    if any(abs(confidence - t) < 1e-6 for t in THRESHOLDS) or not 1 / vocab < confidence < 1:
        raise RuntimeError("bad test input: a confidence on a threshold or out of range")
    shift = float(rng.uniform(-5.0, 5.0))  # softmax ignores a common shift
    logits = [math.log((1.0 - confidence) / (vocab - 1)) + shift] * vocab
    logits[int(rng.integers(vocab))] = math.log(confidence) + shift
    return logits


def phased(rng, phases, vocab=4):
    """A trace made of phases (length, match rate, confidence on a match, on a mismatch)."""
    trace = []
    for length, rate, on_match, on_mismatch in phases:
        for _ in range(length):
            match = bool(rng.random() < rate)
            center = on_match if match else on_mismatch
            confidence = float(np.clip(center + rng.normal(0.0, 0.08), 0.27, 0.99))
            confidence += 2e-6 if any(abs(confidence - t) < 1e-6 for t in THRESHOLDS) else 0.0
            trace.append((match, logits_with_confidence(rng, confidence, vocab)))
    return trace


EASY = (60, 0.97, 0.92, 0.50)  # copy-like text: the draft is almost always right, and sure
HARD = (40, 0.25, 0.60, 0.33)  # the draft is usually wrong, and usually unsure


def check_simulate(ctx, trace, k, schedule, threshold, c_draft, c_target, expected=None):
    args = (trace, k, schedule, threshold, c_draft, c_target)
    got = ctx.call("simulate", *args)
    want = ctx.expected("simulate", *args)
    if expected is not None and expected != want:  # an authoring error
        raise RuntimeError(f"the literal {expected} disagrees with the oracle {want}")
    check.that(isinstance(got, dict), f"simulate: returned a {type(got).__name__}, not a dict")
    check.equal(sorted(got), sorted(want), label="keys of the result")
    detail = (
        f"call: simulate(trace, k={k}, schedule={schedule!r}, threshold={threshold}, "
        f"c_draft={c_draft}, c_target={c_target}) with {len(trace)} positions\n"
        f"matches = {''.join('T' if m else 'F' for m, _ in trace[:80])}"
    )
    for key in ("ks", "rounds", "draft_tokens", "cost"):
        check.that(
            type(got[key]) is type(want[key]),
            f"simulate: result[{key!r}] is a {type(got[key]).__name__}, which is the wrong type",
            f"expected a {type(want[key]).__name__}",
        )
        if key == "ks":
            check.equal(got[key], want[key], label="ks (the draft length of every round)")
        else:
            check.that(
                got[key] == want[key],
                f"simulate: wrong {key}",
                f"{detail}\nexpected {key} = {want[key]}, got {got[key]}\n"
                f"expected ks = {want['ks'][:40]}\ngot      ks = {got['ks'][:40]}",
            )
    return got


def check_stop(ctx, logits, threshold, want):
    got = ctx.call("should_stop_drafting", logits, threshold)
    check.that(
        type(got) is bool,
        f"should_stop_drafting: returned a {type(got).__name__}, not a bool",
    )
    check.that(
        got is want,
        f"should_stop_drafting: returned {got} for a confidence on the other side of the threshold",
        f"logits = {logits}, threshold = {threshold}: expected {want}",
    )


def check_next_k(ctx, k, n_accepted, n_drafted, want):
    got = ctx.call("next_k", k, n_accepted, n_drafted)
    check.that(type(got) is int, f"next_k: returned a {type(got).__name__}, not an int")
    check.that(
        got == want,
        f"next_k({k}, {n_accepted}, {n_drafted}): returned {got}, which is not the next length",
        f"expected {want}",
    )


@case("the worked example: ten positions, k = 2, threshold 0.4", public=True)
def worked_example(ctx):
    check_stop(ctx, [3.0, 0.0], 0.4, False)
    check_stop(ctx, [0.0, 0.0, 0.0], 0.4, True)
    check_next_k(ctx, 2, 2, 2, 4)
    check_next_k(ctx, 4, 0, 3, 3)
    expected = {"cost": 59, "rounds": 5, "draft_tokens": 9, "ks": [2, 4, 3, 2, 4]}
    check_simulate(ctx, WORKED, 2, "heuristic", 0.4, 1, 10, expected)


# Rejects confidence_on_logits, naive_softmax, and stops_on_equality.
@case("should_stop_drafting: a softmax probability, strictly below the threshold", public=True)
def stop_rule(ctx):
    check_stop(ctx, [2.0, 0.0], 0.9, True)  # confidence 0.8808
    check_stop(ctx, [2.0, 0.0], 0.85, False)
    check_stop(ctx, [-1.0, -3.0, -3.0, -3.0], 0.5, False)  # 0.711, with negative logits
    check_stop(ctx, [7.5, 7.0, 7.0, 6.5], 0.4, True)  # 0.375, with large logits
    check_stop(ctx, [1000.0, 998.0], 0.9, True)  # the same 0.8808: only differences count
    check_stop(ctx, [-2000.0, -1000.0, -3000.0], 0.99, False)  # confidence 1.0
    check_stop(ctx, [0.0, 0.0], 0.5, False)  # exactly 0.5 is not below 0.5
    check_stop(ctx, [4.0, 4.0, 4.0, 4.0], 0.25, False)
    check_stop(ctx, [0.3], 0.0, False)
    check_stop(ctx, [0.1, 0.2, 0.3], 0.0, False)  # a threshold of 0 never stops


# Rejects grows_on_any_acceptance, grows_by_one, and shrinks_to_zero.
@case("next_k: +2 when every drafted token was accepted, otherwise -1 down to 1", public=True)
def next_k_table(ctx):
    table = [
        (5, 5, 5, 7),
        (5, 4, 5, 4),
        (5, 0, 5, 4),
        (5, 3, 3, 7),  # a draft cut short at 3 tokens, all accepted
        (5, 2, 3, 4),
        (1, 1, 1, 3),
        (1, 0, 1, 1),  # never below 1
        (2, 1, 2, 1),
        (2, 0, 1, 1),
        (30, 30, 30, 32),  # no upper limit
    ]
    for k, n_accepted, n_drafted, want in table:
        check_next_k(ctx, k, n_accepted, n_drafted, want)


# Rejects no_target_token, counts_later_matches, and target_cost_per_token.
@case("a constant schedule without confidence stopping", public=True)
def constant_schedule(ctx):
    expected = {"cost": 58, "rounds": 5, "draft_tokens": 8, "ks": [2, 2, 2, 2, 2]}
    check_simulate(ctx, WORKED, 2, "constant", 0.0, 1, 10, expected)
    check_simulate(ctx, WORKED, 3, "constant", 0.0, 1, 10)
    check_simulate(ctx, WORKED, 4, "constant", 0.0, 3, 7)
    trace = [(m, [1.0, 0.0]) for m in (T, F, T, T, F, F, T, T, T, F, T, T)]
    check_simulate(ctx, trace, 3, "constant", 0.0, 2, 9)
    check_simulate(ctx, trace, 1, "constant", 0.0, 2, 9)


# Rejects discards_low_confidence_token and compares_with_k.
@case("a low-confidence token ends the draft, is kept, and can still be accepted", public=True)
def confidence_stop(ctx):
    sure, unsure = [3.0, 0.0], [0.0, 0.0, 0.0]
    # Every token matches; every third one is unsure. Drafts are cut to 3 tokens, all
    # accepted, so k keeps growing although no round ever drafts k tokens.
    trace = [(T, unsure if t % 3 == 2 else sure) for t in range(24)]
    got = check_simulate(ctx, trace, 5, "heuristic", 0.4, 1, 10)
    check.that(got["ks"][:3] == [5, 7, 9], "simulate: wrong ks", f"got ks = {got['ks']}")
    check_simulate(ctx, trace, 5, "constant", 0.4, 1, 10)
    # The unsure tokens are the mismatches: each draft ends with a rejected token.
    trace = [(t % 4 != 3, unsure if t % 4 == 3 else sure) for t in range(24)]
    check_simulate(ctx, trace, 6, "heuristic", 0.4, 1, 10)
    check_simulate(ctx, trace, 6, "heuristic", 0.0, 1, 10)


# Rejects never_shrinks_below_initial.
@case("the adaptive policy costs at most 0.9 of the best constant draft length", public=True)
def adaptive_beats_constant(ctx):
    adaptive_cost(ctx, phased(ctx.rng, [EASY, HARD] * 3), k=5)


def adaptive_cost(ctx, trace, k):
    constant = [ctx.expected("simulate", trace, n, "constant", 0.0, 1, 10) for n in range(1, 33)]
    best = min(result["cost"] for result in constant)
    got = check_simulate(ctx, trace, k, "heuristic", 0.4, 1, 10)
    check.at_most("cost", got["cost"], (9 * best) // 10, unit="units")


# Rejects drafts_past_the_end.
@case("the end of the output: never draft the token the target will supply")
def end_of_output(ctx):
    sure = [3.0, 0.0]
    for length in (1, 2, 3, 6, 7):
        trace = [(T, sure)] * length
        for k in (1, 4, 8):
            check_simulate(ctx, trace, k, "heuristic", 0.0, 1, 10)
            check_simulate(ctx, trace, k, "constant", 0.4, 1, 10)
    got = check_simulate(ctx, [(T, sure)] * 6, 8, "constant", 0.0, 1, 10)
    check.that(got["draft_tokens"] == 5, "simulate: wrong draft_tokens")


@case("a draft that is always wrong: k falls to 1 and stays there")
def always_wrong(ctx):
    trace = [(F, [1.0, 0.0])] * 20
    got = check_simulate(ctx, trace, 6, "heuristic", 0.0, 1, 10)
    check.that(min(got["ks"]) >= 1, "simulate: a draft length below 1")
    check_simulate(ctx, trace, 1, "heuristic", 0.0, 1, 10)
    check_simulate(ctx, [(T, [1.0, 0.0])] * 40, 1, "heuristic", 0.0, 1, 10)


@case("phased traces: numeric results for both schedules and several thresholds")
def phased_numeric(ctx):
    for _ in range(3):
        trace = phased(ctx.rng, [HARD, EASY, HARD, (30, 0.6, 0.8, 0.45)])
        for threshold in (0.0, 0.4, 0.7):
            check_simulate(ctx, trace, 3, "heuristic", threshold, 1, 10)
            check_simulate(ctx, trace, 8, "constant", threshold, 2, 15)
        check_simulate(ctx, trace, 1, "heuristic", 0.25, 3, 10)


@case("the cost limit on three more phased traces, from k = 1 and k = 8")
def adaptive_beats_constant_hidden(ctx):
    adaptive_cost(ctx, phased(ctx.rng, [HARD, EASY] * 3), k=1)
    adaptive_cost(ctx, phased(ctx.rng, [EASY, HARD, HARD, EASY, EASY]), k=8)
    adaptive_cost(ctx, phased(ctx.rng, [(25, 0.97, 0.92, 0.5), (25, 0.25, 0.6, 0.33)] * 5), k=5)


@case("random traces, vocabularies, costs, and thresholds")
def random_sweep(ctx):
    for _ in range(20):
        length = int(ctx.rng.integers(1, 60))
        vocab = int(ctx.rng.integers(2, 9))
        rate = float(ctx.rng.uniform(0.1, 0.95))
        trace = []
        for _ in range(length):
            confidence = float(ctx.rng.uniform(1 / vocab + 0.01, 0.99))
            confidence += 2e-6 if any(abs(confidence - t) < 1e-6 for t in THRESHOLDS) else 0.0
            match = bool(ctx.rng.random() < rate)
            trace.append((match, logits_with_confidence(ctx.rng, confidence, vocab)))
        k = int(ctx.rng.integers(1, 10))
        schedule = "heuristic" if ctx.rng.random() < 0.7 else "constant"
        threshold = float(ctx.rng.choice(THRESHOLDS))
        check_simulate(ctx, trace, k, schedule, threshold, *map(int, ctx.rng.integers(1, 30, 2)))
