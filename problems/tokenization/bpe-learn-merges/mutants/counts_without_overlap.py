# BUG: counts only the occurrences a merge would replace (non-overlapping, left to right), so the run a a a counts (a, a) once instead of twice.
from collections import Counter
from itertools import pairwise

END = "</w>"


def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    words = [([*word, END], freq) for word, freq in corpus_words.items()]
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in words:
            previous = None  # the position of the last counted occurrence of a pair (x, x)
            for i, pair in enumerate(pairwise(symbols)):
                if pair[0] == pair[1] and previous == (i - 1, pair):
                    previous = None
                    continue
                counts[pair] += freq
                previous = (i, pair)
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
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
