"""Tests for bpe-encode-fast.

Short inputs are compared with `definition`, a direct transcription of the five steps
of the statement (one scan of the sequence per applied merge), so the expected ids do
not depend on the oracle's data structures. Long inputs are compared with the oracle.
The speed case uses a table in which almost every merge applies exactly once, the
input on which one scan per merge is quadratic.
"""

import string
from collections import Counter
from functools import cache
from itertools import pairwise

import numpy as np

from whetstone.judge import case, check

A, B, C, D = ord("a"), ord("b"), ord("c"), ord("d")

ENGLISH = [
    *["the", "model", "reads", "the", "prompt", "and", "then", "the", "model", "writes"],
    *["the", "next", "token", "streaming", "tokens", "to", "the", "client", "as", "they"],
    *["are", "sampled", "from", "the", "logits", "batching", "requests", "together"],
    *["raises", "throughput", "and", "lowers", "the", "cost", "per", "token", "banana"],
    *["bandana", "cabana", "aaaa", "aaa", "aa", "mississippi", "tennessee", "committee"],
]
WORLD = [
    *["naïve", "café", "señor", "straße", "Ångström", "façade", "jalapeño"],
    *["日本語", "東京", "東京都", "こんにちは", "世界", "你好", "你好吗"],
    *["안녕하세요", "привет", "мир", "Ελληνικά"],
    *["🙂", "🙂🙂", "🚀", "👩\u200d💻", "e\u0301", "a\u0300"],
    *["token", "tokens", "the", "model"],
]

SPEED_BYTES = 2000


def replace_pair(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def definition(text, merges):
    """The statement's steps 1 to 5, literally: one scan per applied merge."""
    rank = {tuple(pair): i for i, pair in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = min(pairwise(ids), key=lambda p: rank.get(p, len(rank)))
        if pair not in rank:
            break
        ids = replace_pair(ids, pair, 256 + rank[pair])
    return ids


def train(data, num_merges):
    """Classic BPE training: repeatedly merge the most frequent adjacent pair."""
    ids, merges = list(data), []
    while len(merges) < num_merges:
        counts = Counter(pairwise(ids))
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        if counts[pair] < 2:
            break
        ids = replace_pair(ids, pair, 256 + len(merges))
        merges.append(pair)
    return merges


def shuffle_ranks(merges, rng):
    """The same vocabulary with the merges in a random priority order.

    Ids are renumbered to follow the new positions, so the table stays well formed,
    but a merge may now refer to a token that a later merge creates.
    """
    position = [int(p) for p in rng.permutation(len(merges))]

    def renumber(token):
        return token if token < 256 else 256 + position[token - 256]

    shuffled = [None] * len(merges)
    for old, (left, right) in enumerate(merges):
        shuffled[position[old]] = (renumber(left), renumber(right))
    return shuffled


def letters(rng, alphabet, n):
    return "".join(str(ch) for ch in rng.choice(list(alphabet), size=n))


def words(rng, pool, n, sep=""):
    return sep.join(str(w) for w in rng.choice(pool, size=n))


def deep_table(rng, n_bytes):
    """A text and a table in which almost every merge applies exactly once.

    The text is cut into blocks of 2 to 48 bytes, and each block is merged down to one
    token in a random bracketing; every pair that is new becomes the next merge. The
    table is as long as the text, and encoding takes about one merge per rank: one
    scan of the sequence per applied merge is then quadratic in the length of the text.
    """
    text = letters(rng, string.ascii_letters + string.digits, n_bytes)
    data = text.encode()
    merges, known, start = [], {}, 0
    while start < len(data):
        size = int(rng.integers(2, 49))
        block = list(data[start : start + size])
        start += size
        while len(block) > 1:
            i = int(rng.integers(len(block) - 1))
            pair = (block[i], block[i + 1])
            if pair not in known:
                known[pair] = 256 + len(merges)
                merges.append(pair)
            block[i : i + 2] = [known[pair]]
    return text, merges


@cache
def tables():
    """Trained tables, built once: English, multilingual, and a small alphabet."""
    rng = np.random.default_rng(20261)
    return {
        "english": train(words(rng, ENGLISH, 300, " ").encode(), 150),
        "world": train(words(rng, WORLD + ENGLISH[:12], 300, " ").encode(), 150),
        "abc": train(letters(rng, "abc", 400).encode(), 60),
    }


def check_encode(ctx, text, merges, want=None, label=None, show=False):
    """Compare with the definition (short texts) or the oracle, and with `want`.

    `show=True` puts the text into the failure message; public cases only.
    """
    label = label or (f"encode({text!r})" if show and len(text) <= 40 else "encode")
    got = ctx.call("encode", text, merges)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{label}: the result must be a list of Python ints",
        f"got: {got!r}"[:400],
    )
    if want is not None:
        check.equal(got, want, label=label)
    if len(text) <= 400:
        check.equal(got, definition(text, merges), label=label)
    else:
        check.equal(got, ctx.expected("encode", text, merges), label=label)
    return got


@case("the worked example: 'aaabdaaabac' with three merges", public=True)
def worked_example(ctx):
    merges = [(A, A), (A, B), (256, 257)]
    check_encode(ctx, "aaabdaaabac", merges, want=[258, 100, 258, 97, 99], show=True)
    check_encode(ctx, "abc", [(B, C), (A, B)], want=[97, 256], show=True)
    check_encode(ctx, "abc", [(A, B), (B, C)], want=[256, 99], show=True)


# Rejects rightmost_first.
@case("runs of one byte merge left to right without overlap", public=True)
def runs(ctx):
    check_encode(ctx, "aaa", [(A, A)], want=[256, 97], show=True)
    check_encode(ctx, "aaaa", [(A, A)], want=[256, 256], show=True)
    check_encode(ctx, "aaaaa", [(A, A), (256, A)], want=[256, 257], show=True)
    merges = [(A, A), (256, 256), (256, A)]
    for n in range(1, 14):
        check_encode(ctx, "a" * n, merges, label=f"encode of {n} a's")
    check_encode(ctx, "baaab aaaaab", merges, show=True)


# Rejects stale_entries_applied: the (a, b) at the start is gone once (b, c) has merged.
@case("a pair that an earlier merge destroyed is not merged", public=True)
def destroyed_pairs(ctx):
    check_encode(ctx, "abc", [(B, C), (A, B)], want=[97, 256], show=True)
    check_encode(ctx, "abcd", [(B, C), (A, B), (C, D)], want=[97, 256, 100], show=True)
    check_encode(ctx, "abab", [(B, A), (A, B)], want=[97, 256, 98], show=True)
    check_encode(ctx, "abcabcab", [(B, C), (A, B), (C, A), (256, A)], show=True)


# Rejects forgets_left_neighbor.
@case("a new token can merge with its left neighbor and with its right neighbor", public=True)
def both_neighbors(ctx):
    check_encode(ctx, "abc", [(A, B), (256, C)], want=[257], show=True)
    check_encode(ctx, "cab", [(A, B), (C, 256)], want=[257], show=True)
    check_encode(ctx, "abab", [(A, B), (256, 256)], want=[257], show=True)
    check_encode(ctx, "dcabab", [(A, B), (C, 256), (D, 257), (258, 256)], want=[259], show=True)


# Rejects eager_new_pairs: in 'abab', merging the first (a, b) creates the pair
# (257, a), which has rank 0, but the pass over (a, b) is not finished.
@case("a new pair of lower rank waits until the current merge is applied everywhere", public=True)
def lower_rank_waits(ctx):
    check_encode(ctx, "abab", [(257, A), (A, B)], want=[257, 257], show=True)
    check_encode(ctx, "abc", [(257, C), (A, B)], want=[256], show=True)
    check_encode(ctx, "ababab", [(257, A), (A, B)], want=[257, 257, 257], show=True)
    check_encode(ctx, "abcabab", [(257, A), (A, B), (257, C)], want=[258, 257, 257], show=True)


# Rejects quadratic_rescan and rescans_to_replace.
@case(
    f"speed: within 5x of the reference on {SPEED_BYTES} bytes with a table as deep as the text",
    public=True,
)
def speed(ctx):
    text, merges = deep_table(ctx.rng, SPEED_BYTES)
    got = ctx.call("encode", text, merges)
    check.equal(got, ctx.expected("encode", text, merges), label="encode")
    ctx.check_speed("encode", text, merges, max_ratio=5.0, rounds=5, warmup=1)


@case("empty text, a single byte, an empty table, and non-ASCII text")
def tiny_inputs(ctx):
    merges = [(A, A), (A, B)]
    check_encode(ctx, "", merges, want=[])
    check_encode(ctx, "", [], want=[])
    check_encode(ctx, "a", merges, want=[97])
    check_encode(ctx, "ab", merges, want=[257])
    check_encode(ctx, "token 🙂", [], want=list("token 🙂".encode()))
    check_encode(ctx, "é", [], want=[195, 169])
    check_encode(ctx, "héé", [(195, 169), (104, 256)], want=[257, 256])
    check_encode(ctx, "日本語 🙂", [(230, 151), (256, 165), (240, 159)])


@case("the arguments are not modified")
def arguments_unchanged(ctx):
    merges = [(A, A), (A, B), (256, 257)]
    before = list(merges)
    got = ctx.solution.encode("aaabdaaabac", merges)
    check.that(merges == before, "encode modified `merges`")
    check.equal(got, [258, 100, 258, 97, 99], label="encode('aaabdaaabac')")


@case("random texts over three letters with a trained table: many runs and repeats")
def small_alphabet(ctx):
    merges = tables()["abc"]
    for _ in range(60):
        check_encode(ctx, letters(ctx.rng, "abc", int(ctx.rng.integers(1, 40))), merges)
    check_encode(ctx, letters(ctx.rng, "ab", 300), merges)


@case("trained tables whose ranks were shuffled: merges that refer to later merges")
def shuffled_ranks(ctx):
    for name in ("abc", "abc", "abc", "world"):
        merges = shuffle_ranks(tables()[name], ctx.rng)
        for _ in range(25):
            if name == "abc":
                text = letters(ctx.rng, "abc", int(ctx.rng.integers(1, 40)))
            else:
                text = words(ctx.rng, WORLD, int(ctx.rng.integers(1, 8)), " ")
            check_encode(ctx, text, merges)


@case("random tables over two letters, in random rank order")
def random_tables(ctx):
    for _ in range(40):
        merges = train(letters(ctx.rng, "ab", 60).encode(), int(ctx.rng.integers(1, 12)))
        merges = shuffle_ranks(merges, ctx.rng)
        for _ in range(5):
            check_encode(ctx, letters(ctx.rng, "ab", int(ctx.rng.integers(1, 30))), merges)


@case("English and multilingual sentences with trained 150-merge tables")
def trained_tables(ctx):
    for _ in range(8):
        check_encode(
            ctx, words(ctx.rng, ENGLISH, int(ctx.rng.integers(1, 25)), " "), tables()["english"]
        )
        check_encode(
            ctx, words(ctx.rng, WORLD, int(ctx.rng.integers(1, 25)), " "), tables()["world"]
        )
    check_encode(ctx, "\x00\x7f\u0080߿ࠀ￿\U00010000\U0010ffff", tables()["world"])


@case("one word of 50,000 bytes, and a run of 49,999 equal bytes")
def long_word(ctx):
    text = words(ctx.rng, ENGLISH + WORLD, 12000)
    text = text.encode()[:50000].decode("utf-8", errors="ignore")
    check_encode(ctx, text, tables()["world"])
    check_encode(ctx, text, shuffle_ranks(tables()["english"], ctx.rng))
    doubling = [(A, A), *[(256 + i, 256 + i) for i in range(14)]]
    check_encode(ctx, "a" * 49999, doubling)
