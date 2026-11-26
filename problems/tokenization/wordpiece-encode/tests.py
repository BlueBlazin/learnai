"""Tests for wordpiece-encode.

The statement's examples are compared with literal ids; everything else with the
oracle, on hand-written vocabularies and on random ones built here from a seeded
generator. Vocabularies are passed as fresh copies, and ids are checked to be ints.
"""

from learnai.judge import case, check

UNK = "[UNK]"

EXAMPLE_TOKENS = ["[UNK]", "un", "##aff", "##able", "aff", "able", "##a", "u", "##n", "##ly"]
EXAMPLE = {tok: i for i, tok in enumerate([*EXAMPLE_TOKENS, "abc", "ab", "##cd"])}

# A small English-like vocabulary: every letter as a word-initial piece, most letters
# as continuation pieces (no "##q", "##x", "##z"), and common pieces of both kinds.
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ENGLISH = {tok: i for i, tok in enumerate([UNK, "[CLS]", "[SEP]", *LETTERS])}
for _tok in [
    *["##" + c for c in LETTERS if c not in "qxz"],
    *["the", "token", "tokens", "embed", "rank", "re", "un", "in", "play", "work"],
    *["##ing", "##ed", "##er", "##s", "##ly", "##ness", "##ization", "##ize", "##tion"],
    *["##token", "##rank", "##play", "##work", "##able", "##ding", "##in", "##ion"],
]:
    ENGLISH[_tok] = len(ENGLISH)

WORLD = {
    tok: 100 + i
    for i, tok in enumerate(
        [UNK, "na", "##ï", "##ve", "naï", "caf", "##é", "日本", "##語", "日", "##本", "東", "##京"]
        + ["🙂", "##🙂", "e", "##́", "é", "#", "##", "###", "##a", "a", "##b", "###b"]
    )
}


def encode(ctx, word, vocab, max_chars=100, unk=UNK, want=None):
    """Compare with the oracle (and with `want`); the result must be a list of ints."""
    before = dict(vocab)
    got = ctx.call("wordpiece", word, dict(vocab), unk, max_chars)
    check.that(
        isinstance(got, list) and all(type(i) is int for i in got),
        "wordpiece must return a list of Python ints",
        f"got {got!r}",
    )
    label = f"wordpiece({word!r})"
    if want is not None:
        check.equal(got, want, label=label)
    check.equal(got, ctx.expected("wordpiece", word, dict(before), unk, max_chars), label=label)
    return got


def random_vocab(rng, alphabet, n_pieces, drop):
    """Single characters (some dropped) plus random multi-character pieces of both kinds."""
    tokens = [UNK]
    for c in alphabet:
        if rng.random() >= drop:
            tokens.append(c)
        if rng.random() >= drop:
            tokens.append("##" + c)
    for _ in range(n_pieces):
        piece = "".join(rng.choice(list(alphabet), size=int(rng.integers(2, 6))))
        tokens.append(piece if rng.random() < 0.4 else "##" + piece)
    tokens = list(dict.fromkeys(tokens))
    ids = rng.permutation(len(tokens))
    return {tok: int(i) for tok, i in zip(tokens, ids, strict=True)}


def random_word(rng, alphabet, low, high):
    return "".join(rng.choice(list(alphabet), size=int(rng.integers(low, high))))


@case("the worked example: unaffable, able, unable, unaffably, abcd", public=True)
def worked_example(ctx):
    encode(ctx, "unaffable", EXAMPLE, want=[1, 2, 3])
    encode(ctx, "able", EXAMPLE, want=[5])
    encode(ctx, "unable", EXAMPLE, want=[1, 3])
    encode(ctx, "unaffably", EXAMPLE, want=[0])
    encode(ctx, "abcd", EXAMPLE, want=[0])
    encode(ctx, "unaffable", EXAMPLE, max_chars=8, want=[0])


# Rejects shortest_match, prefix_on_first_piece, and no_continuation_prefix.
@case("longest match first, with ## on every piece after the first", public=True)
def longest_match(ctx):
    v = ENGLISH
    encode(ctx, "tokens", v, want=[v["tokens"]])
    encode(ctx, "tokenization", v, want=[v["token"], v["##ization"]])
    encode(ctx, "reranking", v, want=[v["re"], v["##rank"], v["##ing"]])
    encode(ctx, "embedding", v, want=[v["embed"], v["##ding"]])
    encode(ctx, "ing", v, want=[v["in"], v["##g"]])  # "##ing" is not a first piece
    encode(ctx, "replay", v, want=[v["re"], v["##play"]])
    encode(ctx, "playplay", v, want=[v["play"], v["##play"]])


# Rejects unk_for_failing_piece and backtracks.
@case("a word that cannot be finished is one unknown token, with no backtracking", public=True)
def whole_word_unknown(ctx):
    v = ENGLISH
    encode(ctx, "tokenizer", v, want=[v["token"], v["##ize"], v["##r"]])
    encode(ctx, "tokenz", v, want=[v[UNK]])  # token, then "##z" is missing
    encode(ctx, "relax", v, want=[v[UNK]])  # re ##l ##a, then "##x" is missing
    encode(ctx, "x", v, want=[v["x"]])  # word-initial "x" exists
    encode(ctx, "xx", v, want=[v[UNK]])
    encode(ctx, "Token", v, want=[v[UNK]])  # no case folding here
    backtrack = {UNK: 0, "abc": 1, "ab": 2, "##cd": 3, "a": 4, "##b": 5, "##c": 6, "##bcd": 7}
    encode(ctx, "abcd", backtrack, want=[0])
    encode(ctx, "abc", backtrack, want=[1])


# Rejects max_chars_off_by_one and no_length_limit.
@case("max_chars: a word of exactly max_chars is split, one character more is not", public=True)
def length_limit(ctx):
    v = ENGLISH
    encode(ctx, "tokens", v, max_chars=6, want=[v["tokens"]])
    encode(ctx, "tokens", v, max_chars=5, want=[v[UNK]])
    encode(ctx, "a" * 12, v, max_chars=12, want=[v["a"]] + [v["##a"]] * 11)
    encode(ctx, "a" * 13, v, max_chars=12, want=[v[UNK]])
    encode(ctx, "a", v, max_chars=1, want=[v["a"]])
    encode(ctx, "", v, max_chars=1, want=[])


# Rejects counts_bytes: "日本語" is 3 characters and 9 bytes.
@case("characters are code points: non-ASCII words, and words containing #")
def unicode_words(ctx):
    v = WORLD
    encode(ctx, "naïve", v, max_chars=5, want=[v["naï"], v["##ve"]])
    encode(ctx, "café", v, max_chars=4, want=[v["caf"], v["##é"]])
    encode(ctx, "日本語", v, max_chars=3, want=[v["日本"], v["##語"]])
    encode(ctx, "東京", v, max_chars=2, want=[v["東"], v["##京"]])
    encode(ctx, "🙂🙂", v, max_chars=2, want=[v["🙂"], v["##🙂"]])
    encode(ctx, "é", v, max_chars=2, want=[v["e"], v["##́"]])  # decomposed é
    encode(ctx, "é", v, max_chars=1, want=[v["é"]])
    encode(ctx, "日本語", v, max_chars=2, want=[v[UNK]])
    # "#" is an ordinary character of the word; only candidates get a prefix added.
    encode(ctx, "##a", v, want=[v["##a"]])
    encode(ctx, "#b", v, want=[v["#"], v["##b"]])
    encode(ctx, "a#b", v, want=[v["a"], v["###b"]])
    encode(ctx, "##", v, want=[v["##"]])


@case("the unknown token and its id are whatever the caller passes")
def other_unknown_tokens(ctx):
    vocab = {"<unk>": 7, "[UNK]": 3, "a": 1, "##b": 2}
    encode(ctx, "ab", vocab, unk="<unk>", want=[1, 2])
    encode(ctx, "ba", vocab, unk="<unk>", want=[7])
    encode(ctx, "ba", vocab, unk="[UNK]", want=[3])
    encode(ctx, "abab", vocab, max_chars=3, unk="<unk>", want=[7])
    # The unknown token's own text is an ordinary word.
    encode(ctx, "[UNK]", {"[UNK]": 0, "[": 1, "##U": 2, "##N": 3, "##K": 4, "##]": 5}, want=[0])
    encode(ctx, "<unk>", {"<unk>": 9, "<": 1, "##unk": 2, "##>": 3}, unk="<unk>", want=[9])


@case("random vocabularies over a small alphabet: every word splits")
def random_complete_vocab(ctx):
    for _ in range(6):
        vocab = random_vocab(ctx.rng, "abcd", n_pieces=40, drop=0.0)
        for _ in range(40):
            word = random_word(ctx.rng, "abcd", 1, 16)
            got = encode(ctx, word, vocab)
            check.that(
                vocab[UNK] not in got, "wordpiece: got the unknown token", f"word = {word!r}"
            )


@case("random vocabularies with missing characters: some words are unknown")
def random_incomplete_vocab(ctx):
    for _ in range(6):
        vocab = random_vocab(ctx.rng, "abcdef", n_pieces=60, drop=0.25)
        for _ in range(60):
            encode(ctx, random_word(ctx.rng, "abcdef", 1, 12), vocab)


@case("random words around the length limit, including the longest allowed")
def random_lengths(ctx):
    vocab = random_vocab(ctx.rng, "abc", n_pieces=30, drop=0.0)
    for max_chars in (1, 2, 7, 50, 200):
        for delta in (-1, 0, 1, 2):
            n = max(max_chars + delta, 0)
            word = random_word(ctx.rng, "abc", n, n + 1)
            encode(ctx, word, vocab, max_chars=max_chars)
    encode(ctx, random_word(ctx.rng, "abc", 400, 401), vocab, max_chars=200)
