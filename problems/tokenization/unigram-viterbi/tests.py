"""Tests for unigram-viterbi.

Short inputs are compared with `brute_force`, which enumerates every segmentation and
applies the statement's ordering directly, so the expected pieces do not depend on the
oracle's dynamic program. Long inputs are compared with the oracle. Every score is a
multiple of 1/8, so sums are exact in floating point and ties are real ties.
"""

from itertools import combinations, pairwise

from learnai.judge import case, check

HUG = {"h": -3.0, "u": -3.0, "g": -3.0, "s": -2.0, "hu": -4.0, "ug": -3.5, "gs": -4.5, "hug": -9.0}


def unk_score(pieces):
    return min(pieces.values()) - 10.0


def piece_score(piece, pieces):
    """The score of one piece of a segmentation, or None if it is not allowed."""
    if piece in pieces:
        return pieces[piece]
    return unk_score(pieces) if len(piece) == 1 else None


def brute_force(text, pieces):
    """Every segmentation, ranked as the statement says. For short texts only."""
    n = len(text)
    if n == 0:
        return [], 0.0
    best_key, best = None, None
    for k in range(n):
        for cuts in combinations(range(1, n), k):
            bounds = [0, *cuts, n]
            parts = [text[a:b] for a, b in pairwise(bounds)]
            scores = [piece_score(part, pieces) for part in parts]
            if any(score is None for score in scores):
                continue
            # Highest score, then fewest pieces, then the longest earlier pieces.
            key = (sum(scores), -len(parts), tuple(len(part) for part in parts))
            if best_key is None or key > best_key:
                best_key, best = key, parts
    return best, best_key[0]


def total_score(parts, pieces):
    return sum(piece_score(part, pieces) for part in parts)


def encode(ctx, text, pieces, want=None, show=False):
    """Structure, then the brute force (short texts) or the oracle, then `want`.

    `show=True` puts the text into the failure message; public cases only.
    """
    label = f"unigram_encode({text!r})" if show and len(text) <= 24 else "unigram_encode"
    given = dict(pieces)
    got = ctx.solution.unigram_encode(text, given)
    check.that(given == pieces, f"{label}: `pieces` was modified")
    check.that(
        isinstance(got, list) and all(isinstance(piece, str) for piece in got),
        f"{label}: the result must be a list of strings",
        f"got: {got!r}"[:400],
    )
    check.that(
        "".join(got) == text,
        f"{label}: the pieces do not concatenate to the text",
        f"text: {text!r}\ngot:  {got!r}"[:800],
    )
    check.that(
        all(piece_score(piece, pieces) is not None for piece in got),
        f"{label}: a returned piece is neither in the vocabulary nor a single character",
        f"got: {got!r}"[:400],
    )
    if want is not None:
        check.equal(got, want, label=label)
    if len(text) <= 13:
        expected, score = brute_force(text, pieces)
        check.that(
            total_score(got, pieces) >= score,
            f"{label}: another segmentation has a higher total score",
            f"got:    {got!r} with score {total_score(got, pieces)}\n"
            f"better: {expected!r} with score {score}",
        )
        check.equal(got, expected, label=label)
    else:
        check.equal(got, ctx.oracle.unigram_encode(text, dict(pieces)), label=label)
    return got


def random_vocab(rng, alphabet, n_pieces, max_len, scores, singles=0.8):
    """Random pieces over `alphabet`; each single character is included with
    probability `singles`, so some characters are unknown."""
    pieces = {}
    for ch in alphabet:
        if rng.random() < singles:
            pieces[ch] = float(rng.choice(scores))
    while len(pieces) < n_pieces:
        length = int(rng.integers(2, max_len + 1))
        piece = "".join(str(ch) for ch in rng.choice(list(alphabet), size=length))
        pieces[piece] = float(rng.choice(scores))
    return pieces


def random_text(rng, alphabet, n):
    return "".join(str(ch) for ch in rng.choice(list(alphabet), size=n))


EIGHTHS = [-(k / 8) for k in range(4, 80)]  # -0.5 to -9.875 in steps of 1/8


# Rejects greedy_longest_match: the longest piece at the start is "hug".
@case("the worked example: 'hugs' and 'hug' with the eight-piece vocabulary", public=True)
def worked_example(ctx):
    encode(ctx, "hugs", HUG, want=["hu", "gs"], show=True)
    encode(ctx, "hug", HUG, want=["h", "ug"], show=True)
    encode(ctx, "hugz", HUG, want=["h", "ug", "z"], show=True)


# Rejects ties_ignore_piece_count and ties_later_longer.
@case("ties: fewer pieces first, then the longer earlier piece", public=True)
def ties(ctx):
    two = {"a": -1.0, "c": -1.0, "ab": -2.0, "bc": -2.0}
    encode(ctx, "abc", two, want=["ab", "c"], show=True)
    fewer = {"a": -2.0, "c": -1.0, "d": -1.0, "ab": -1.0, "bcd": -1.0}
    encode(ctx, "abcd", fewer, want=["a", "bcd"], show=True)
    later = {"a": -1.0, "b": -1.0, "c": -1.0, "d": -1.0, "ab": -2.0, "bc": -2.0, "cd": -2.0}
    encode(ctx, "abcd", later, want=["ab", "cd"], show=True)
    encode(ctx, "aaaaa", {"a": -1.0, "aa": -2.0, "aaa": -3.0}, want=["aaa", "aa"], show=True)
    encode(ctx, "aaaa", {"a": -1.0, "aa": -2.0, "aaa": -3.0}, want=["aaa", "a"], show=True)


# Rejects no_unk_penalty, unk_penalty_from_zero, and unk_only_when_no_piece_matches.
@case("unknown characters: one piece each, at the lowest score minus 10", public=True)
def unknown_characters(ctx):
    encode(ctx, "xyz", {"a": -1.0}, want=["x", "y", "z"], show=True)
    # 'a' is unknown at -11.5 - 10: "ab" at -11.5 beats ?a + b at -22.5.
    encode(ctx, "ab", {"b": -1.0, "ab": -11.5}, want=["ab"], show=True)
    # Without the penalty, ?a + bc at -5 would beat ab + c at -8.
    encode(ctx, "abc", {"ab": -4.0, "c": -4.0, "bc": -1.0}, want=["ab", "c"], show=True)
    # A longer piece starts at 'a', and the unknown 'a' is still the better choice.
    encode(ctx, "abcd", {"ab": -1.0, "bcd": -1.0}, want=["a", "bcd"], show=True)
    encode(ctx, "abab", {"ab": -1.0, "ba": -1.0, "b": -1.0}, want=["ab", "ab"], show=True)
    encode(ctx, "a b", {"a": -1.0, "b": -1.0}, want=["a", " ", "b"], show=True)


# Rejects byte_positions.
@case("positions are characters: accented letters, CJK, and emoji", public=True)
def characters_not_bytes(ctx):
    french = {"é": -2.0, "t": -2.0, "ét": -3.0, "té": -3.5}
    encode(ctx, "été", french, want=["ét", "é"], show=True)
    encode(ctx, "日本語", {"日本": -1.0, "語": -3.0, "本語": -1.5}, want=["日本", "語"], show=True)
    encode(ctx, "日本語", {"日本": -1.0, "本語": -1.5}, want=["日本", "語"], show=True)
    encode(ctx, "日本語", {"日本": -1.5, "本語": -1.0}, want=["日", "本語"], show=True)
    encode(ctx, "a🙂b", {"a": -1.0, "b": -1.0}, want=["a", "🙂", "b"], show=True)
    encode(ctx, "né🙂", {"n": -1.0, "e": -1.0, "né": -1.0}, want=["né", "🙂"], show=True)


@case("no segmentation scores higher: random short texts against every segmentation", public=True)
def no_better_segmentation(ctx):
    def prop(rng):
        alphabet = "abcd"
        pieces = random_vocab(rng, alphabet, int(rng.integers(3, 14)), 4, EIGHTHS)
        text = random_text(rng, alphabet, int(rng.integers(1, 12)))
        encode(ctx, text, pieces, show=True)

    check.forall(prop, trials=40)


@case("empty text, one character, and a text that is one piece")
def tiny_inputs(ctx):
    encode(ctx, "", HUG, want=[])
    encode(ctx, "h", HUG, want=["h"])
    encode(ctx, "z", HUG, want=["z"])
    encode(ctx, "hug", {"hug": -9.0}, want=["hug"])
    encode(ctx, "hu", {"hug": -9.0}, want=["h", "u"])
    encode(ctx, "x", {"x": 0.0}, want=["x"])


@case("random short texts where most segmentations tie: scores are -1 or -2")
def many_ties(ctx):
    for _ in range(80):
        pieces = random_vocab(ctx.rng, "ab", int(ctx.rng.integers(2, 9)), 4, [-1.0, -2.0], 0.9)
        encode(ctx, random_text(ctx.rng, "ab", int(ctx.rng.integers(1, 13))), pieces)
    for _ in range(40):
        pieces = random_vocab(ctx.rng, "abc", int(ctx.rng.integers(2, 12)), 5, [-1.0, -1.5, -3.0])
        encode(ctx, random_text(ctx.rng, "abc", int(ctx.rng.integers(1, 13))), pieces)


@case("vocabularies with few or no single characters: unknowns between long pieces")
def sparse_singles(ctx):
    for _ in range(80):
        pieces = random_vocab(ctx.rng, "abc", int(ctx.rng.integers(2, 10)), 4, EIGHTHS, 0.3)
        encode(ctx, random_text(ctx.rng, "abc", int(ctx.rng.integers(1, 13))), pieces)
    for _ in range(20):
        pieces = random_vocab(ctx.rng, "éü日本🙂a", int(ctx.rng.integers(2, 12)), 3, EIGHTHS, 0.5)
        encode(ctx, random_text(ctx.rng, "éü日本🙂ab", int(ctx.rng.integers(1, 12))), pieces)


@case("pieces longer than the text, and pieces of up to 24 characters")
def long_pieces(ctx):
    word = "tokenization-and-decoding"
    pieces = {ch: -6.0 for ch in set(word)}
    pieces |= {word[:24]: -8.0, word[5:]: -9.0, word * 2: -1.0, "token": -4.0, "ization": -4.5}
    encode(ctx, word, pieces)
    encode(ctx, word[:24], pieces, want=[word[:24]])
    encode(ctx, word[:12], pieces)
    encode(ctx, word + word[:3], pieces)


@case("long texts: 2,000 characters over a 300-piece vocabulary")
def long_texts(ctx):
    alphabet = "abcdefgh é日"
    pieces = random_vocab(ctx.rng, alphabet, 300, 8, EIGHTHS, 0.8)
    for n in (200, 2000):
        encode(ctx, random_text(ctx.rng, alphabet + "z", n), pieces)
    ties = random_vocab(ctx.rng, "abc", 40, 6, [-1.0, -2.0, -3.0], 1.0)
    encode(ctx, random_text(ctx.rng, "abc", 2000), ties)
