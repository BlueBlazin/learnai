"""Tests for prompt-lookup-draft.

Everything is exact. The worked example is compared with literal values; every other
call is compared with `definition`, a deliberately different formulation of the same
rule (collect every earlier occurrence, then take the minimum), and with the oracle.
"""

from learnai.judge import case, check

FN = "lookup_draft"


def definition(context, max_ngram, min_ngram, k):
    length = len(context)
    for n in range(max_ngram, min_ngram - 1, -1):
        starts = [
            s for s in range(length - n) if context[s : s + n] == context[length - n :]
        ]
        if n < length and starts:
            return context[min(starts) + n : min(starts) + n + k]
    return []


def lookup(ctx, context, max_ngram, min_ngram, k, expected=None):
    given = list(context)
    got = ctx.call(FN, given, max_ngram, min_ngram, k)
    want = definition(list(context), max_ngram, min_ngram, k)
    if expected is not None and expected != want:  # an authoring error
        raise RuntimeError("the literal expectation disagrees with the definition")
    if want != ctx.expected(FN, list(context), max_ngram, min_ngram, k):  # an authoring error
        raise RuntimeError("the reference disagrees with the definition")
    shown = context if len(context) <= 40 else f"{context[:40]} ... ({len(context)} tokens)"
    detail = (
        f"call: {FN}(context, max_ngram={max_ngram}, min_ngram={min_ngram}, k={k})\n"
        f"  context = {shown}\nexpected {want}\ngot {got!r}"
    )
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{FN}: must return a list of Python ints",
        detail,
    )
    check.that(len(got) <= k, f"{FN}: returned more than k tokens", detail)
    check.that(got == want, f"{FN}: wrong draft", detail)
    check.that(given == list(context), f"{FN}: the context was modified", detail)
    check.that(got is not given, f"{FN}: must return a new list, not the context itself")
    return got


@case("the worked example: earliest match, a short draft, an overlap, no match", public=True)
def worked_example(ctx):
    lookup(ctx, [5, 9, 2, 7, 4, 9, 2, 8, 3, 9, 2], 3, 1, 3, expected=[7, 4, 9])
    lookup(ctx, [1, 2, 3, 1, 2], 3, 1, 4, expected=[3, 1, 2])
    lookup(ctx, [7, 7, 7], 2, 1, 4, expected=[7])
    lookup(ctx, [4, 6, 5, 8, 6], 3, 2, 4, expected=[])


@case("an n-gram that occurs three times: the earliest occurrence is used", public=True)
def earliest(ctx):
    # Rejects most_recent_occurrence. Each occurrence of [3, 4] continues differently.
    context = [3, 4, 10, 11, 12, 0, 3, 4, 20, 21, 22, 0, 3, 4, 30, 31, 32, 1, 3, 4]
    lookup(ctx, context, 2, 2, 3, expected=[10, 11, 12])
    lookup(ctx, context, 4, 1, 2, expected=[10, 11])


@case("a longer n-gram wins over a shorter one, even if the shorter occurs earlier", public=True)
def longest_first(ctx):
    # Rejects shortest_first: [9] alone occurs first at index 0; [8, 9] at index 4.
    context = [9, 50, 51, 5, 8, 9, 40, 41, 6, 8, 9]
    lookup(ctx, context, 3, 1, 2, expected=[40, 41])
    lookup(ctx, context, 1, 1, 2, expected=[50, 51])


@case("nothing repeats, or the context is too short: the draft is empty", public=True)
def no_match(ctx):
    # A correct solution returns a non-empty draft elsewhere; matches_itself never does.
    lookup(ctx, [1, 2, 3, 4, 5, 6], 3, 1, 5, expected=[])
    lookup(ctx, [], 3, 1, 5, expected=[])
    lookup(ctx, [9], 3, 1, 5, expected=[])
    lookup(ctx, [9, 9], 3, 1, 5, expected=[9])


@case("min_ngram is respected: a match that exists only below it gives no draft")
def min_ngram_respected(ctx):
    # Rejects ignores_min_ngram.
    context = [2, 5, 7, 1, 3, 5, 9, 4, 6, 5]
    lookup(ctx, context, 3, 2, 4, expected=[])
    lookup(ctx, context, 3, 1, 4, expected=[7, 1, 3, 5])
    lookup(ctx, [1, 2, 3, 9, 2, 3], 4, 3, 2, expected=[])
    lookup(ctx, [1, 2, 3, 9, 1, 2, 3], 4, 3, 2, expected=[9, 1])


@case("occurrences that overlap the suffix: runs of one token and short cycles")
def overlapping(ctx):
    # Rejects no_overlap, and matches_itself on inputs where an occurrence exists.
    lookup(ctx, [7] * 3, 3, 1, 4, expected=[7])
    lookup(ctx, [7] * 10, 3, 1, 4, expected=[7, 7, 7, 7])
    lookup(ctx, [5, 7, 7, 7, 7], 3, 3, 4, expected=[7])
    lookup(ctx, [1, 2, 1, 2, 1, 2, 1], 4, 1, 6, expected=[2, 1])
    lookup(ctx, [0, 3, 3, 4, 3, 3, 4, 3, 3], 5, 2, 8, expected=[4, 3, 3])


@case("fewer than k tokens follow the earliest occurrence: a shorter draft, not another match")
def short_continuation(ctx):
    # Rejects requires_full_draft.
    context = [6, 1, 40, 6, 1]
    lookup(ctx, context, 2, 1, 4, expected=[40, 6, 1])
    lookup(ctx, context, 2, 1, 1, expected=[40])
    lookup(ctx, context, 2, 1, 64, expected=[40, 6, 1])


@case("the copy starts right after the match: k = 1 and distinct neighbors")
def copy_offset(ctx):
    # Rejects copies_from_match_end.
    context = [11, 12, 13, 14, 15, 16, 17, 12, 13]
    lookup(ctx, context, 2, 1, 1, expected=[14])
    lookup(ctx, context, 2, 1, 3, expected=[14, 15, 16])
    lookup(ctx, context, 8, 1, 3, expected=[14, 15, 16])


@case("max_ngram larger than the context, and a suffix that is the whole context")
def max_ngram_exceeds_length(ctx):
    lookup(ctx, [4, 4], 8, 1, 3, expected=[4])
    lookup(ctx, [4, 5, 4, 5], 8, 4, 3, expected=[])
    lookup(ctx, [4, 5, 4, 5], 8, 2, 3, expected=[4, 5])


@case("random contexts over small vocabularies, every parameter varied")
def random_contexts(ctx):
    rng = ctx.rng
    drafts = 0
    for _ in range(300):
        vocab = int(rng.integers(2, 7))
        context = [int(t) for t in rng.integers(0, vocab, size=int(rng.integers(0, 40)))]
        min_ngram = int(rng.integers(1, 5))
        max_ngram = int(rng.integers(min_ngram, 9))
        drafts += bool(lookup(ctx, context, max_ngram, min_ngram, int(rng.integers(1, 9))))
    if not 60 < drafts < 280:  # an authoring error
        raise RuntimeError("the random contexts no longer mix hits and misses")


@case("a 4096-token context that quotes itself, and one with no repeat at all")
def long_context(ctx):
    rng = ctx.rng
    document = [int(t) for t in rng.integers(5, 64, size=3000)]
    quote = document[1200:1500]
    context = [0, *document, 2, *quote, 3, *document[700:780], 4, *quote[:14]]
    context += [5] * (4096 - len(context) - 14) + quote[:14]
    lookup(ctx, context, 8, 1, 64)
    lookup(ctx, context, 3, 2, 10)
    lookup(ctx, list(range(4096)), 8, 1, 64, expected=[])
