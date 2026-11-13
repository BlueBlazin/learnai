Byte-pair encoding builds a vocabulary by repeatedly merging the most frequent adjacent pair of symbols. Training starts from individual bytes and stops after a fixed number of merges; the ordered list of merges **is** the tokenizer.

Implement the training loop.

## Interface

```python
def train_bpe(words: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """
    words: word -> frequency in the corpus.
    Returns the merges in the order they were learned.
    """
```

Each word starts as a sequence of single characters. Pairs never span word boundaries.

## One iteration

1. Count every adjacent pair across all words, weighted by word frequency.
2. Pick the pair with the highest count. Break ties by choosing the lexicographically smallest pair.
3. Replace every non-overlapping occurrence of that pair, left to right, with the merged symbol.

### Example 1

```python
train_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, num_merges=3)
# [("e", "s"), ("es", "t"), ("l", "o")]
```

| Merge | Pair counts that decided it | Vocabulary gains |
|---:|---|---|
| 1 | `("e","s")` 9, `("s","t")` 9, `("w","e")` 8 | `es` |
| 2 | `("es","t")` 9, `("l","o")` 7 | `est` |
| 3 | `("l","o")` 7, `("o","w")` 7 | `lo` |

In the first round `("e","s")` and `("s","t")` tie at 9; `("e","s")` is lexicographically smaller.

## Properties the judge checks

- The result has exactly `num_merges` entries, or fewer if no pairs remain.
- Applying the merges in order to the training words reproduces the symbol sequences of the reference.
- Overlapping occurrences are handled left to right: in `aaa` the pair `("a","a")` merges once, giving `aa a`.

## Constraints

- Up to $20{,}000$ distinct words, word length up to 40, $1 \le$ `num_merges` $\le 2000$.
- Pure Python; the standard library only.
