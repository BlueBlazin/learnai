# BUG: counts the pairs once, on the initial characters, and returns the n most frequent; the counts must be taken again after every merge.
from collections import Counter
from itertools import pairwise

END = "</w>"


def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    words = [([*word, END], freq) for word, freq in corpus_words.items()]
    counts: Counter[tuple[str, str]] = Counter()
    for symbols, freq in words:
        for pair in pairwise(symbols):
            counts[pair] += freq
    return sorted(counts, key=lambda pair: (-counts[pair], pair))[:n_merges]


def _merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    out = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out
