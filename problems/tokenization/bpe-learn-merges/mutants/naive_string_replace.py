# BUG: applies a merge with str.replace on the space-joined symbols, which also matches the end of a longer symbol followed by the start of another ("xa b" contains "a b").
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
    return " ".join(symbols).replace(" ".join(pair), "".join(pair)).split(" ")
