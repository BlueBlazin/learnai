from collections import Counter
from itertools import pairwise

END = "</w>"


def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    """Learn a BPE merge table from word frequencies.

    corpus_words: word -> number of occurrences in the corpus (a positive int).
    n_merges: the number of merges to learn.
    Each word starts as its characters followed by the symbol "</w>". Repeatedly: count
    every adjacent pair of symbols, weighted by word frequency; take the most frequent
    pair (ties: the smallest pair in tuple order); merge it in every word, left to
    right without overlap.
    Returns the merges in the order they were learned, as (left, right) tuples of
    strings; fewer than n_merges if no pair is left.
    """
    raise NotImplementedError
