"""Tests for suffix-automaton-draft.

Everything but the speed case is exact. `definition` is the statement written as
plainly as possible (try every suffix length from the longest down, scan every start
index); every numeric case replays a sequence token by token and compares the
solution's `match_length()` and `draft(...)` with it after each `extend`, and, as a
check on the reference itself, with the oracle. On the long sequences the definition
is too slow to run at every step, so the oracle stands in for it there, after the
shorter cases have tied the two together.

The speed case drives the loop a decoder would: one `extend` and one `draft` per token.
"""

from learnai.judge import case, check

CLS = "SuffixIndex"


def definition(seq, max_len, min_match):
    """(match length, draft) for the sequence `seq`, by exhaustive search."""
    size = len(seq)
    for n in range(size - 1, 0, -1):
        for start in range(size - n):
            if seq[start : start + n] == seq[size - n :]:
                draft = seq[start + n : start + n + max_len] if n >= min_match else []
                return n, draft
    return 0, []


def shown(seq):
    return seq if len(seq) <= 40 else f"[... {len(seq)} tokens ending with {seq[-12:]}]"


def compare(index, seq, max_len, min_match, want_length, want_draft):
    detail = (
        f"after extend() with each of {shown(seq)}\n"
        f"expected match_length() = {want_length}, "
        f"draft({max_len}, {min_match}) = {want_draft}"
    )
    got_length = index.match_length()
    check.that(
        type(got_length) is int and got_length == want_length,
        f"match_length: wrong match length after {len(seq)} tokens",
        f"{detail}\ngot match_length() = {got_length!r}",
    )
    got = index.draft(max_len, min_match)
    detail += f"\ngot draft = {got!r}"
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        "draft: must return a list of Python ints",
        detail,
    )
    check.that(len(got) <= max_len, "draft: returned more than max_len tokens", detail)
    check.that(got == want_draft, f"draft: wrong draft after {len(seq)} tokens", detail)
    return got


def replay(ctx, tokens, queries, *, brute=True, every=1, expected=None):
    """Feed `tokens` one by one; after every `every`-th, check each (max_len, min_match).

    Returns the list of match lengths (one per checked step).
    """
    index = ctx.solution.SuffixIndex()
    oracle = ctx.oracle.SuffixIndex()
    seq, lengths = [], []
    for step, token in enumerate(tokens):
        check.that(
            index.extend(token) is None, "extend: must return None", f"got a value at step {step}"
        )
        oracle.extend(token)
        seq.append(token)
        if (step + 1) % every and step + 1 != len(tokens):
            continue
        for max_len, min_match in queries:
            want = (oracle.match_length(), oracle.draft(max_len, min_match))
            if brute and want != definition(seq, max_len, min_match):  # an authoring error
                raise RuntimeError("the reference disagrees with the definition")
            compare(index, seq, max_len, min_match, *want)
        lengths.append(oracle.match_length())
    if expected is not None and lengths != expected:  # an authoring error
        raise RuntimeError("the literal expectation disagrees with the definition")
    return lengths


def final(ctx, tokens, max_len, min_match, expected):
    """Feed all of `tokens`, then check one draft against a literal."""
    index = ctx.solution.SuffixIndex()
    for token in tokens:
        index.extend(token)
    want = definition(list(tokens), max_len, min_match)
    if want[1] != expected:  # an authoring error
        raise RuntimeError("the literal expectation disagrees with the definition")
    compare(index, list(tokens), max_len, min_match, *want)


EXAMPLE = [5, 9, 2, 7, 4, 9, 2, 8, 3, 9, 2]


@case("the worked example: match lengths after every token, then three drafts", public=True)
def worked_example(ctx):
    replay(ctx, EXAMPLE, [(3, 1)], expected=[0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2])
    final(ctx, EXAMPLE, 3, 1, [7, 4, 9])
    final(ctx, EXAMPLE, 3, 2, [7, 4, 9])
    final(ctx, EXAMPLE, 3, 3, [])
    final(ctx, [7, 7, 7], 4, 1, [7])


@case("an empty index, a single token, and a sequence in which nothing repeats", public=True)
def nothing_to_match(ctx):
    index = ctx.solution.SuffixIndex()
    compare(index, [], 4, 1, 0, [])
    replay(ctx, [3], [(4, 1)], expected=[0])
    replay(ctx, list(range(40, 60)), [(4, 1)], expected=[0] * 20)


@case("the match is the longest suffix seen before, however long", public=True)
def longest_match(ctx):
    # Rejects shortest_usable_match: [9] alone occurs first at index 0, [8, 9] at 4.
    final(ctx, [9, 50, 51, 5, 8, 9, 40, 41, 6, 8, 9], 2, 1, [40, 41])
    # A 12-token quote: the match grows by one per token and never stops at a bound.
    quote = list(range(20, 32))
    lengths = replay(ctx, [*quote, 1, 2, *quote], [(4, 1), (4, 6)])
    if lengths[-12:] != list(range(1, 13)):  # an authoring error
        raise RuntimeError("the quote is not matched token by token")


@case("the earliest occurrence of the match is the one copied from", public=True)
def earliest(ctx):
    # Rejects most_recent_occurrence and clone_ends_here. Each occurrence of [3, 4]
    # continues differently.
    seq = [3, 4, 10, 11, 12, 0, 3, 4, 20, 21, 22, 0, 3, 4, 30, 31, 32, 1, 3, 4]
    final(ctx, seq, 3, 1, [10, 11, 12])
    final(ctx, seq, 3, 2, [10, 11, 12])
    # The earliest occurrence of the match [5, 6], not of its shorter suffix [6].
    final(ctx, [6, 70, 5, 6, 80, 5, 6, 90, 5, 6], 2, 1, [80, 5])
    # The state of [1, 2] is split off a longer one when 2 is seen after 9: its
    # earliest end is still index 1.
    final(ctx, [1, 2, 3, 1, 2, 4, 9, 2, 5, 1, 2], 3, 2, [3, 1, 2])
    replay(ctx, [1, 2, 3, 1, 2, 4, 9, 2, 5, 1, 2], [(3, 1), (3, 2)])


@case("min_match and max_len: a match of exactly min_match drafts; the copy stops at the end",
      public=True)  # fmt: skip
def thresholds(ctx):
    # Rejects min_match_exclusive and copies_from_match_end.
    seq = [11, 12, 13, 14, 15, 16, 17, 12, 13]
    final(ctx, seq, 1, 2, [14])
    final(ctx, seq, 3, 2, [14, 15, 16])
    final(ctx, seq, 3, 3, [])
    final(ctx, seq, 64, 1, [14, 15, 16, 17, 12, 13])
    final(ctx, [6, 1, 40, 6, 1], 4, 2, [40, 6, 1])
    final(ctx, [6, 1, 40, 6, 1], 1, 1, [40])


@case("speed: one extend and one draft per token over 12,000 tokens, within 5x of the reference",
      public=True, time_limit_s=30)  # fmt: skip
def online_speed(ctx):
    # Rejects rescans_history. Copy-heavy text over a small vocabulary: long matches,
    # many split states.
    rng = ctx.rng
    document = [int(t) for t in rng.integers(2, 40, size=3_000)]
    tokens = list(document)
    while len(tokens) < 12_000:
        start = int(rng.integers(0, len(document) - 400))
        tokens += document[start : start + int(rng.integers(20, 400))]
        tokens += [int(t) for t in rng.integers(2, 40, size=int(rng.integers(1, 12)))]
    tokens = tokens[:12_000]
    replay(ctx, tokens[:1500], [(8, 2)], brute=False, every=7)  # correct on this input first

    def loop(cls):
        def run():
            index = cls()
            drafted = 0
            for token in tokens:
                index.extend(token)
                drafted += len(index.draft(8, 2))
            return drafted

        return run

    ctx.compare_speed(
        loop(ctx.solution.SuffixIndex),
        loop(ctx.reference.SuffixIndex),
        max_ratio=5.0,
        label="12,000 extend + draft steps",
        rounds=3,
        warmup=1,
        tokens=len(tokens),
    )


@case("matches that overlap themselves: runs of one token and short cycles")
def overlapping(ctx):
    replay(ctx, [7] * 12, [(4, 1), (4, 3)], expected=list(range(12)))
    final(ctx, [7] * 10, 4, 1, [7])
    replay(ctx, [1, 2] * 8, [(6, 1), (3, 4)])
    replay(ctx, [0, 3, 3, 4, 3, 3, 4, 3, 3], [(8, 2)])
    replay(ctx, [5, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7], [(5, 1)])


@case("random sequences over two to five tokens, checked after every token")
def random_small_vocab(ctx):
    # Small vocabularies make nearly every extend split a state: rejects
    # clone_without_transitions and never_clones on many inputs.
    rng = ctx.rng
    nonempty = 0
    for _ in range(60):
        vocab = int(rng.integers(2, 6))
        tokens = [int(t) for t in rng.integers(0, vocab, size=int(rng.integers(1, 60)))]
        queries = [(int(rng.integers(1, 9)), int(rng.integers(1, 5)))]
        lengths = replay(ctx, tokens, queries)
        nonempty += sum(1 for n in lengths if n >= queries[0][1])
    if nonempty < 400:  # an authoring error
        raise RuntimeError("the random sequences produce too few drafts")


@case("token ids up to 100,000 and a sequence that quotes itself at several distances")
def large_ids(ctx):
    rng = ctx.rng
    document = [int(t) for t in rng.integers(0, 100_000, size=120)]
    tokens = [*document, 7, *document[30:70], 8, *document[10:50], 9, *document[35:60]]
    replay(ctx, tokens, [(16, 1), (5, 8)], every=4)


@case("two indexes are independent, and draft() does not change the index")
def independent_instances(ctx):
    first, second = ctx.solution.SuffixIndex(), ctx.solution.SuffixIndex()
    seq_a, seq_b = [4, 5, 6, 4, 5], [9, 9, 4, 5, 8, 9, 4]
    for token in seq_a:
        first.extend(token)
    for token in seq_b:
        second.extend(token)
    for _ in range(2):  # the second round sees whatever the first round left behind
        got = compare(first, seq_a, 3, 1, *definition(seq_a, 3, 1))
        got.append(-1)  # the result is the caller's: changing it must not reach the index
        compare(second, seq_b, 3, 1, *definition(seq_b, 3, 1))
    first.extend(6)
    compare(first, [*seq_a, 6], 3, 1, *definition([*seq_a, 6], 3, 1))


@case("a 4,096-token context that quotes a document, checked every 13 tokens")
def long_context(ctx):
    rng = ctx.rng
    document = [int(t) for t in rng.integers(5, 64, size=2500)]
    quote = document[1200:1500]
    tokens = [0, *document, 2, *quote, 3, *document[700:780], 4, *quote[:140]]
    tokens += [int(t) for t in rng.integers(5, 9, size=4096 - len(tokens))]
    replay(ctx, tokens, [(64, 1), (10, 3)], brute=False, every=13)
    final(ctx, tokens[:300], 10, 1, definition(tokens[:300], 10, 1)[1])
