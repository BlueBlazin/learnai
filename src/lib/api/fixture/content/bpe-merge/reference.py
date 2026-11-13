from collections import Counter


def merge_word(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(words: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    corpus = {tuple(word): count for word, count in words.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for symbols, count in corpus.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] += count
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        corpus = {merge_word(symbols, best): count for symbols, count in corpus.items()}
    return merges
