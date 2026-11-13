"""Tests for bpe-encode.

Worked examples are compared with literal ids; everything else with the oracle, plus
two property cases that state what encoding means without reference to the oracle.
Merge tables are either written by hand or trained here on small seeded corpora.
"""

from collections import Counter
from functools import cache
from itertools import pairwise

import numpy as np

from whetstone.judge import case, check

A, B, C = ord("a"), ord("b"), ord("c")

# Word pools for the training corpora and the test sentences.
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


def sentence(rng, words, n):
    return " ".join(str(w) for w in rng.choice(words, size=n))


@cache
def tables():
    """Three trained tables, built once: English, multilingual, and rank-shuffled."""
    rng = np.random.default_rng(20240)
    english = train(sentence(rng, ENGLISH, 400).encode(), 120)
    world = train(sentence(rng, WORLD + ENGLISH[:12], 400).encode(), 120)
    shuffled = shuffle_ranks(world, rng)
    return {"english": english, "world": world, "shuffled": shuffled}


def token_bytes(ids, merges):
    """The test's own expansion of token ids to bytes."""
    data, stack = bytearray(), list(reversed(ids))
    while stack:
        token = stack.pop()
        if token < 256:
            data.append(token)
        else:
            stack.extend(reversed(merges[token - 256]))
    return bytes(data)


def random_text(rng, length):
    """Words from both pools mixed with random code points from several scripts."""
    ranges = [(0x20, 0x7F), (0xA0, 0x100), (0x370, 0x500), (0x4E00, 0x4E80), (0x1F600, 0x1F650)]
    words = ENGLISH + WORLD
    parts = []
    for _ in range(length):
        if rng.random() < 0.5:
            parts.append(str(rng.choice(words)))
        else:
            low, high = ranges[int(rng.integers(len(ranges)))]
            parts.append(chr(int(rng.integers(low, high))))
        if rng.random() < 0.3:
            parts.append(" ")
    return "".join(parts)


def check_encode(ctx, text, merges, label="encode"):
    got = ctx.call("encode", text, merges)
    check.equal(got, ctx.expected("encode", text, merges), label=label)


@case("the worked example: 'aaabdaaabac' with three merges", public=True)
def worked_example(ctx):
    merges = [(A, A), (A, B), (256, 257)]
    ids = ctx.call("encode", "aaabdaaabac", merges)
    check.equal(ids, [258, 100, 258, 97, 99], label="encode('aaabdaaabac')")
    check.equal(ctx.call("decode", [258, 100, 258, 97, 99], merges), "aaabdaaabac", label="decode")


@case("rank decides, not position: 'abc' with (b,c) ranked before (a,b)", public=True)
def rank_beats_position(ctx):
    # Rejects leftmost_pair_first.
    check.equal(ctx.call("encode", "abc", [(B, C), (A, B)]), [97, 256], label="encode('abc')")
    check.equal(ctx.call("encode", "abc", [(A, B), (B, C)]), [256, 99], label="encode('abc')")
    check_encode(ctx, "abcbc abab cbcab", [(B, C), (A, B), (C, 256), (257, 257)])


@case("non-ASCII text is encoded from its UTF-8 bytes", public=True)
def utf8_bytes(ctx):
    # Rejects chars_not_bytes.
    check.equal(ctx.call("encode", "é", []), [195, 169], label="encode('é')")
    merges = [(195, 169), (104, 256)]
    check.equal(ctx.call("encode", "héé", merges), [257, 256], label="encode('héé')")
    check.equal(ctx.call("decode", [257, 256], merges), "héé", label="decode")
    check_encode(ctx, "日本語 🙂", [(230, 151), (256, 165), (240, 159)], label="encode('日本語 🙂')")


@case("runs of one byte merge left to right without overlap", public=True)
def runs(ctx):
    # Rejects overlapping_merges.
    check.equal(ctx.call("encode", "aaa", [(A, A)]), [256, 97], label="encode('aaa')")
    check.equal(ctx.call("encode", "aaaa", [(A, A)]), [256, 256], label="encode('aaaa')")
    merges = [(A, A), (256, 256), (256, A)]
    for n in range(1, 12):
        check_encode(ctx, "a" * n, merges, label=f"encode of {n} a's")
    check_encode(ctx, "baaab aaaaab", merges, label="encode('baaab aaaaab')")


@case("a merge that refers to a token created by a later merge", public=True)
def forward_reference(ctx):
    # Rejects list_order_single_sweep.
    merges = [(257, C), (A, B)]
    check.equal(ctx.call("encode", "abc", merges), [256], label="encode('abc')")
    check.equal(ctx.call("decode", [256], merges), "abc", label="decode")
    check_encode(ctx, "ababc cabc abcabc", merges, label="encode('ababc cabc abcabc')")


@case("empty text, a single byte, and an empty merge table")
def tiny_inputs(ctx):
    merges = [(A, A), (A, B)]
    check.equal(ctx.call("encode", "", merges), [], label="encode('')")
    check.equal(ctx.call("encode", "a", merges), [97], label="encode('a')")
    check.equal(ctx.call("encode", "ab", merges), [257], label="encode('ab')")
    check.equal(ctx.call("encode", "token 🙂", []), list("token 🙂".encode()), label="no merges")
    check.equal(ctx.call("decode", [], merges), "", label="decode([])")
    check.equal(ctx.call("decode", [], []), "", label="decode([])")


@case("a longer token exists in the vocabulary but the merge order never builds it")
def not_longest_match(ctx):
    # Rejects greedy_longest_match: 'abc' is token 258, yet (b,c) merges first.
    merges = [(B, C), (A, B), (257, C)]
    check_encode(ctx, "abc", merges, label="encode('abc')")
    check_encode(ctx, "abcab abc cabcb", merges, label="encode('abcab abc cabcb')")
    merges = [(A, B), (C, ord("d")), (B, C), (256, 257), (A, 258)]
    for text in ["abcd", "abc", "bcd", "abcdabcabcd", "aabcdd"]:
        check_encode(ctx, text, merges, label=f"encode({text!r})")


@case("English sentences with a trained 120-merge table")
def trained_english(ctx):
    merges = tables()["english"]
    for _ in range(12):
        text = sentence(ctx.rng, ENGLISH, int(ctx.rng.integers(1, 25)))
        check_encode(ctx, text, merges)
    check_encode(ctx, "Unseen WORDS, digits 0123456789 and punctuation!?", merges)


@case("multilingual text with a trained table: accents, CJK, emoji, combining marks")
def trained_multilingual(ctx):
    merges = tables()["world"]
    for _ in range(12):
        check_encode(ctx, random_text(ctx.rng, int(ctx.rng.integers(1, 30))), merges)
    check_encode(ctx, "\x00\x7f\u0080߿ࠀ￿\U00010000\U0010ffff", merges)


@case("trained tables whose ranks were shuffled")
def shuffled_ranks(ctx):
    for _ in range(4):
        merges = shuffle_ranks(tables()["world"], ctx.rng)
        for _ in range(4):
            check_encode(ctx, random_text(ctx.rng, int(ctx.rng.integers(1, 30))), merges)


@case("decode: arbitrary ids, characters split across tokens, invalid UTF-8")
def decode_cases(ctx):
    # Rejects decode_per_token: the two bytes of 'é' arrive as separate tokens.
    def check_decode(ids, merges, label):
        got = ctx.call("decode", ids, merges)
        check.that(isinstance(got, str), f"{label}: decode must return a str")
        check.equal(got, ctx.expected("decode", ids, merges), label=label)

    check_decode([104, 195, 169], [], "decode of 'h' and the two bytes of 'é'")
    check_decode([240, 159, 256], [(153, 130)], "decode of an emoji split across tokens")
    check_decode([104, 195], [], "decode of a truncated character")
    check_decode([255, 97, 128, 98], [], "decode of invalid bytes")
    for name, merges in tables().items():
        for _ in range(6):
            ids = [int(t) for t in ctx.rng.integers(0, 256 + len(merges), size=20)]
            check_decode(ids, merges, f"decode of random ids ({name} table)")
            ids = ctx.expected("encode", random_text(ctx.rng, 12), merges)
            check_decode(ids, merges, f"decode of encoded text ({name} table)")


@case("property: decode(encode(s)) == s for random Unicode strings")
def round_trip(ctx):
    def prop(rng):
        name = ("english", "world", "shuffled")[int(rng.integers(3))]
        merges = tables()[name]
        text = random_text(rng, int(rng.integers(0, 40)))
        ids = ctx.call("encode", text, merges)
        back = ctx.call("decode", ids, merges)
        check.that(
            back == text,
            "decode(encode(s)) is not s",
            f"table: {name}\ns       = {text!r}\nencoded = {ids!r}\ndecoded = {back!r}\n",
        )

    check.forall(prop, trials=40)


@case("property: the ids spell the UTF-8 bytes of the text and no merge is left to apply")
def fixed_point(ctx):
    def prop(rng):
        name = ("english", "world", "shuffled")[int(rng.integers(3))]
        merges = tables()[name]
        text = random_text(rng, int(rng.integers(0, 40)))
        ids = ctx.call("encode", text, merges)
        detail = f"table: {name}\ntext    = {text!r}\nencoded = {ids!r}\n"
        check.that(
            isinstance(ids, list)
            and all(type(t) is int and 0 <= t < 256 + len(merges) for t in ids),
            "encode must return a list of ints in range(256 + len(merges))",
            detail,
        )
        check.that(
            token_bytes(ids, merges) == text.encode("utf-8"),
            "the bytes of the returned tokens are not the UTF-8 bytes of the text",
            detail,
        )
        known = set(merges)
        leftover = [pair for pair in pairwise(ids) if pair in known]
        check.that(
            not leftover,
            "the output still contains an adjacent pair that is in the merge table",
            detail + f"mergeable pairs left: {leftover[:5]!r}\n",
        )

    check.forall(prop, trials=40)
