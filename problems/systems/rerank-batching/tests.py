"""Tests for rerank-batching.

Scores are compared with the kit scorer's own (uncounted) score of each truncated
pair, so the expected values do not come from the reference. The cost limits are
what the statement's batching rule produces, computed here by `plan`.
"""

from whetstone.judge import case, check
from whetstone.kit import CountingScorer

FN = "rerank"


def truncated(query, docs, scorer):
    room = scorer.max_length - len(query)
    return [doc[:room] for doc in docs]


def plan(lengths, max_batch, max_tokens):
    """(padded tokens, calls) of the statement's rule for these pair lengths."""
    padded = calls = size = width = 0
    for length in sorted(lengths):
        fits = size and size < max_batch
        if fits and max_tokens is not None:
            fits = (size + 1) * length <= max_tokens
        if not fits:
            padded += size * width
            calls += 1
            size = 0
        size, width = size + 1, length
    return padded + size * width, calls


def run(ctx, query, docs, *, max_batch, top_n, max_tokens=None, max_length=48, seed=1,
        cost=True):  # fmt: skip
    """Call the solution, check scores, ranking, and (with `cost`) the scorer's counters."""
    scorer = CountingScorer(seed=seed, max_length=max_length, max_batch=max_batch)
    before = ([*query], [[*doc] for doc in docs])
    got = ctx.solution.rerank(query, docs, scorer, max_batch, top_n, max_tokens)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        "rerank: must return a tuple (scores, ranking)",
        f"got: {type(got).__name__}",
    )
    scores, ranking = got
    check.that((query, docs) == before, "rerank: the query or a document was modified in place")
    cut = truncated(query, docs, scorer)
    want = [scorer.score(query, doc) for doc in cut]
    check.that(
        isinstance(scores, list) and all(isinstance(s, float) for s in scores),
        "rerank: scores must be a list of floats",
    )
    check.equal(len(scores), len(want), label="scores")
    wrong = [i for i, (g, w) in enumerate(zip(scores, want, strict=True)) if g != w]
    if wrong:
        moved = sorted(scores) == sorted(want)
        check.that(
            False,
            f"scores: {len(wrong)} of {len(want)} documents have a score that is not the "
            "scorer's score of (query, that document)"
            + (": the right scores, in the wrong positions" if moved else ""),
            f"first at document {wrong[0]}: got {scores[wrong[0]]!r}, expected "
            f"{want[wrong[0]]!r}\nexpected scores = {want}\ngot scores = {scores}",
        )
    want_ranking = sorted(range(len(want)), key=lambda i: -want[i])[:top_n]
    check.equal(ranking, want_ranking, label="ranking")
    if cost:
        lengths = [len(query) + len(doc) for doc in cut]
        padded, calls = plan(lengths, max_batch, max_tokens)
        if max_tokens is not None:
            over = [b for b in scorer.batches if len(b) * max(b) > max_tokens]
            check.that(
                not over,
                f"a call held {len(over[0]) * max(over[0]) if over else 0} padded tokens, "
                f"which exceeds max_tokens = {max_tokens}",
                f"pair lengths of that call: {over[0] if over else None}",
            )
        check.cost(scorer.counters, "pairs", exactly=len(docs), unit="pairs")
        check.cost(scorer.counters, "padded_tokens", at_most=padded, unit="tokens")
        check.cost(scorer.counters, "calls", at_most=calls, unit="calls")
    return scorer


def make_docs(rng, lengths):
    return [[int(t) for t in rng.integers(2, 64, size=n)] for n in lengths]


EXAMPLE_QUERY = [7, 3]
EXAMPLE_DOCS = [[5], [9, 9, 9, 9, 9, 9], [4, 8], [6, 6, 6, 6, 6, 6], [2]]


@case("the worked example: 5 documents, max_batch 4, max_tokens 12: 28 tokens in 3 calls",
      public=True)  # fmt: skip
def worked_example(ctx):
    check.that(plan([3, 8, 4, 8, 3], 4, 12) == (28, 3), "test bug")
    scorer = run(ctx, EXAMPLE_QUERY, EXAMPLE_DOCS, max_batch=4, top_n=2, max_tokens=12)
    check.equal(
        sorted(sorted(b) for b in scorer.batches),
        [[3, 3, 4], [8], [8]],
        label="pair lengths of the three calls",
    )


# Rejects arrival_order_batches (40 padded tokens instead of 28).
@case("the worked example without a token budget, max_batch 3: 28 tokens in 2 calls",
      public=True)  # fmt: skip
def worked_example_no_budget(ctx):
    check.that(plan([3, 8, 4, 8, 3], 3, None) == (28, 2), "test bug")
    run(ctx, EXAMPLE_QUERY, EXAMPLE_DOCS, max_batch=3, top_n=5)


# Rejects scores_in_sorted_order: the lengths are in descending order, so the sorted
# order is the reverse of the arrival order.
@case("scores[i] belongs to docs[i]: documents arriving longest first", public=True)
def scores_follow_documents(ctx):
    docs = make_docs(ctx.rng, [30, 24, 19, 13, 8, 5, 2, 1])
    run(ctx, [11, 12, 13], docs, max_batch=3, top_n=3, cost=False)


# Rejects arrival_order_batches, one_call_per_pair, and ignores_token_budget.
@case("40 documents of mixed lengths: padded tokens and calls within the limits", public=True)
def mixed_lengths_cost(ctx):
    rng = ctx.rng
    lengths = [int(n) for n in rng.choice([2, 3, 5, 9, 17, 30, 44], size=40)]
    docs = make_docs(rng, lengths)
    query = [int(t) for t in rng.integers(2, 64, size=4)]
    run(ctx, query, docs, max_batch=8, top_n=5)
    run(ctx, query, docs, max_batch=16, top_n=5, max_tokens=96)


# Rejects truncation_drops_query: half the documents are longer than the scorer takes.
@case("documents longer than the scorer accepts are cut at the end; the query is kept",
      public=True)  # fmt: skip
def truncation(ctx):
    rng = ctx.rng
    docs = make_docs(rng, [3, 40, 12, 29, 64, 21, 22, 5])
    query = [int(t) for t in rng.integers(2, 64, size=6)]
    run(ctx, query, docs, max_batch=4, top_n=4, max_length=24)
    run(ctx, query, docs, max_batch=4, top_n=4, max_length=24, max_tokens=48)


# Rejects empty_batch_call.
@case("no documents; top_n of 0; top_n larger than the number of documents")
def edge_sizes(ctx):
    scorer = run(ctx, [5, 6], [], max_batch=4, top_n=3)
    check.cost(scorer.counters, "calls", exactly=0, unit="calls")
    docs = make_docs(ctx.rng, [4, 9, 2])
    run(ctx, [5, 6], docs, max_batch=4, top_n=0)
    run(ctx, [5, 6], docs, max_batch=4, top_n=10)
    run(ctx, [5, 6], docs[:1], max_batch=1, top_n=1, max_tokens=48)


# Rejects ties_to_higher_index: duplicate documents have exactly equal scores.
@case("duplicate documents: equal scores are ranked by index, lower first")
def stable_ties(ctx):
    rng = ctx.rng
    a, b, c = make_docs(rng, [6, 11, 3])
    docs = [a, b, a, c, b, a, c, b]
    for seed in (1, 2, 3):
        run(ctx, [9, 8, 7], docs, max_batch=3, top_n=8, seed=seed)
        run(ctx, [9, 8, 7], docs, max_batch=3, top_n=4, seed=seed)


# Rejects budget_uses_previous_width: a long pair after two short ones.
@case("the token budget is tested with the pair being added")
def budget_width(ctx):
    docs = make_docs(ctx.rng, [2, 2, 38, 2, 40, 39])
    run(ctx, [1, 2], docs, max_batch=4, top_n=6, max_tokens=48)
    run(ctx, [1, 2], docs, max_batch=6, top_n=6, max_tokens=100)


@case("a batch of one pair at a time when max_batch is 1; equal lengths throughout")
def degenerate_batches(ctx):
    rng = ctx.rng
    docs = make_docs(rng, [7] * 9)
    run(ctx, [3], docs, max_batch=1, top_n=3)
    run(ctx, [3], docs, max_batch=4, top_n=3, max_tokens=64)


@case("random candidate lists, limits, and scorers")
def random_lists(ctx):
    rng = ctx.rng
    for trial in range(12):
        n = int(rng.integers(1, 60))
        max_length = int(rng.choice([16, 32, 48]))
        lengths = [int(x) for x in rng.integers(1, 60, size=n)]
        docs = make_docs(rng, lengths)
        if trial % 3 == 0:  # some duplicates
            docs = [docs[int(j)] for j in rng.integers(0, max(n // 2, 1), size=n)]
        query = [int(t) for t in rng.integers(2, 64, size=int(rng.integers(1, 9)))]
        max_batch = int(rng.integers(1, 12))
        max_tokens = None if trial % 2 else int(rng.integers(max_length, 6 * max_length))
        run(ctx, query, docs, max_batch=max_batch, top_n=int(rng.integers(0, n + 3)),
            max_tokens=max_tokens, max_length=max_length, seed=trial)  # fmt: skip
