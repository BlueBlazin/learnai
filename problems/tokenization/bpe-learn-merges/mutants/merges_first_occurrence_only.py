# BUG: merges only the first occurrence of the pair in each word, so a word that contains the pair twice keeps an unmerged copy.
from collections import Counter
from itertools import pairwise

END = "</w>"


def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    words = [([*word, END], freq) for word, freq in corpus_words.items()]
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in words:
            for pair in pairwise(symbols):
                counts[pair] += freq
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        words = [(_merge(symbols, best), freq) for symbols, freq in words]
    return merges


def _merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    for i in range(len(symbols) - 1):
        if (symbols[i], symbols[i + 1]) == pair:
            return [*symbols[:i], symbols[i] + symbols[i + 1], *symbols[i + 2 :]]
    return symbols
