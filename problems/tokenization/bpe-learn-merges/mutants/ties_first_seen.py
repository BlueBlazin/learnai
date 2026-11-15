# BUG: among pairs with the same count, takes the one counted first (max over the dict), which depends on the order of the corpus instead of being the smallest pair.
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
        best = max(counts, key=lambda pair: counts[pair])
        merges.append(best)
        words = [(_merge(symbols, best), freq) for symbols, freq in words]
    return merges


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
