from collections import Counter
from itertools import pairwise

END = "</w>"


def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    # Each distinct word is a list of symbols: its characters, then the end marker.
    words = [([*word, END], freq) for word, freq in corpus_words.items()]
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        # Count every adjacent pair, weighted by how often the word occurs.
        counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in words:
            for pair in pairwise(symbols):
                counts[pair] += freq
        if not counts:
            break  # every word is a single symbol: nothing left to merge
        # Highest count wins; among equals, the smallest pair in tuple order.
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        words = [(_merge(symbols, best), freq) for symbols, freq in words]
    return merges


def _merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every occurrence of `pair`, scanning left to right without overlap."""
    out = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2  # both halves are consumed, so "a a a" becomes [aa, a]
        else:
            out.append(symbols[i])
            i += 1
    return out
