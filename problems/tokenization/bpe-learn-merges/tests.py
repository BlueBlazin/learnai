"""Tests for bpe-learn-merges.

The statement's examples are compared with literal merge lists, everything else with
the oracle. Corpora are passed as fresh dicts, in several key orders where the order
could tempt an implementation, and are checked to come back unmodified.
"""

from learnai.judge import case, check

PAPER = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
PAPER_MERGES = [
    *[("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w")],
    *[("e", "w"), ("ew", "est</w>"), ("n", "ewest</w>"), ("low", "</w>"), ("d", "est</w>")],
]

ENGLISH = [
    *["the", "model", "reads", "prompt", "and", "then", "writes", "next", "token"],
    *["tokens", "tokenizer", "tokenization", "streaming", "stream", "streams", "client"],
    *["sampled", "sampling", "sampler", "logits", "batching", "batch", "batches", "requests"],
    *["request", "throughput", "lower", "lowest", "newer", "newest", "wider", "widest"],
    *["cache", "cached", "caching", "banana", "bandana", "mississippi", "committee", "aaaa"],
]
WORLD = ["naïve", "café", "señor", "straße", "日本語", "日本", "東京", "東京都", "你好", "你好吗"]
WORLD += ["привет", "мир", "🙂", "🙂🙂", "é", "été", "tête"]


def learn(ctx, corpus, n_merges, want=None):
    """Compare with the oracle (and with `want`); check the structure of the result."""
    given = dict(corpus)
    got = ctx.solution.learn_merges(given, n_merges)
    check.that(given == corpus, "learn_merges modified `corpus_words`")
    check.that(
        isinstance(got, list)
        and all(
            isinstance(pair, (tuple, list))
            and len(pair) == 2
            and all(isinstance(symbol, str) for symbol in pair)
            for pair in got
        ),
        "learn_merges must return a list of (left, right) tuples of strings",
        f"got {got!r}",
    )
    got = [tuple(pair) for pair in got]
    if want is not None:
        check.equal(got, want, label="merges")
    check.equal(got, ctx.oracle.learn_merges(dict(corpus), n_merges), label="merges")
    return got


def reordered(corpus, rng):
    keys = [str(k) for k in rng.permutation(list(corpus))]
    return {k: corpus[k] for k in keys}


def random_corpus(rng, alphabet, n_words, max_len, max_freq):
    words = {}
    for _ in range(n_words):
        word = "".join(rng.choice(list(alphabet), size=int(rng.integers(1, max_len + 1))))
        words[word] = int(rng.integers(1, max_freq + 1))
    return words


@case("the worked example: the four-word corpus of the paper, 10 merges", public=True)
def worked_example(ctx):
    learn(ctx, PAPER, 10, want=PAPER_MERGES)
    learn(ctx, PAPER, 3, want=PAPER_MERGES[:3])
    learn(ctx, PAPER, 6, want=PAPER_MERGES[:6])


# Rejects counts_without_overlap, and merges that are applied with overlap.
@case("a run a a a: both positions are counted, one merge is applied", public=True)
def runs(ctx):
    learn(ctx, {"aaa": 3, "ab": 5}, 2, want=[("a", "a"), ("a", "b")])
    learn(ctx, {"bbba": 3}, 2, want=[("b", "b"), ("a", "</w>")])
    learn(ctx, {"aaaa": 1}, 3, want=[("a", "a"), ("aa", "</w>"), ("aa", "aa</w>")])
    learn(ctx, {"aaaaa": 2, "b": 9}, 4)


# Rejects unweighted_counts.
@case("pairs are weighted by word frequency", public=True)
def weighted_by_frequency(ctx):
    corpus = {"xy": 10, "ab": 1, "ac": 1, "ad": 1}
    learn(ctx, corpus, 3, want=[("x", "y"), ("xy", "</w>"), ("a", "b")])
    learn(ctx, {"ba": 3, "cab": 2}, 4)
    learn(ctx, {"zz": 1000000, "ab": 999999, "abab": 1}, 3)


# Rejects ties_first_seen and ties_largest.
@case("ties go to the smallest pair, whatever the order of the dict", public=True)
def ties(ctx):
    learn(ctx, {"bab": 4}, 1, want=[("a", "b")])
    want = [("a", "b"), ("ab", "</w>"), ("c", "d")]
    learn(ctx, {"cd": 2, "ab": 2}, 3, want=want)
    learn(ctx, {"ab": 2, "cd": 2}, 3, want=want)
    for _ in range(4):
        learn(ctx, reordered(PAPER, ctx.rng), 10, want=PAPER_MERGES)


# Rejects no_end_of_word, marker_glued_to_last_char, stale_counts, stops_at_singletons.
@case("the marker </w> is a symbol, and merging stops only when no pair is left", public=True)
def marker_and_stopping(ctx):
    learn(ctx, {"ba": 4}, 4, want=[("a", "</w>"), ("b", "a</w>")])
    learn(ctx, {"ab": 1}, 10, want=[("a", "b"), ("ab", "</w>")])
    learn(ctx, {"a": 1, "b": 3}, 10, want=[("b", "</w>"), ("a", "</w>")])
    want = [("a", "a"), ("aa", "b"), ("aab", "b"), ("aabb", "</w>")]
    learn(ctx, {"aabb": 1}, 5, want=want)
    learn(ctx, {"aabb": 1}, 4, want=want)


@case("no merges requested, and an empty corpus")
def nothing_to_do(ctx):
    learn(ctx, PAPER, 0, want=[])
    learn(ctx, {}, 5, want=[])
    learn(ctx, {}, 0, want=[])


# Rejects merges_first_occurrence_only and naive_string_replace.
@case("a merge replaces every occurrence in a word, and only whole symbols")
def applying_a_merge(ctx):
    learn(ctx, {"baba": 2}, 2, want=[("b", "a"), ("ba", "</w>")])
    learn(ctx, {"banana": 2, "bandana": 1}, 6)
    learn(ctx, {"a": 1, "aaaa": 1}, 3, want=[("a", "a"), ("a", "</w>"), ("aa", "</w>")])
    learn(ctx, {"abcabcab": 3, "cab": 2, "bca": 2, "b": 4}, 8)
    learn(ctx, {"mississippi": 2, "miss": 3, "sip": 1, "is": 4}, 12)


@case("non-ASCII words: characters are code points")
def unicode_words(ctx):
    corpus = {word: 1 + (i * 7) % 5 for i, word in enumerate(WORLD)}
    learn(ctx, corpus, 25)
    learn(ctx, {"日本語": 2, "日本": 3}, 3, want=[("日", "本"), ("日本", "</w>"), ("日本", "語")])


@case("an English word list: 60, 150, and more merges than there are pairs")
def english(ctx):
    corpus = {
        word: int(f) for word, f in zip(ENGLISH, ctx.rng.zipf(1.3, len(ENGLISH)), strict=True)
    }
    corpus = {word: min(freq, 10**6) for word, freq in corpus.items()}
    learn(ctx, corpus, 60)
    learn(ctx, reordered(corpus, ctx.rng), 150)
    got = learn(ctx, corpus, 300)
    check.that(len(got) < 300, "learn_merges returned merges after every word was one symbol")


@case("random corpora over two and three letters, with many ties and runs")
def random_small_alphabets(ctx):
    for _ in range(60):
        alphabet = "ab" if ctx.rng.random() < 0.5 else "abc"
        corpus = random_corpus(ctx.rng, alphabet, int(ctx.rng.integers(1, 6)), 6, 4)
        learn(ctx, corpus, int(ctx.rng.integers(1, 12)))


@case("random corpora: up to 500 words, skewed frequencies, in shuffled order")
def random_corpora(ctx):
    for n_words, max_len, n_merges in ((20, 8, 30), (80, 12, 100), (500, 30, 60)):
        corpus = random_corpus(ctx.rng, "abcdefgh", n_words, max_len, 1)
        freqs = ctx.rng.zipf(1.5, len(corpus))
        corpus = {w: int(min(f, 10**6)) for w, f in zip(corpus, freqs, strict=True)}
        learn(ctx, reordered(corpus, ctx.rng), n_merges)
