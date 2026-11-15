Implement the procedure that builds a BPE tokenizer's merge table from a corpus:
repeatedly find the most frequent pair of adjacent symbols and fuse it into a new
symbol. This is counting, not model training, and its whole difficulty is determinism:
the table is only reproducible if the counting, the tie-breaking, and the way a merge
is applied are all pinned down.

## Specification

```python
def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]: ...
```

- `corpus_words`: the corpus as word frequencies. Each key is a word, each value the
  number of times it occurs (a positive int). The corpus has already been split into
  words, and merges never cross word boundaries.
- `n_merges`: the number of merges to learn, a non-negative int.
- Returns the learned merges in order, as a list of `(left, right)` tuples of strings.

**Symbols.** Every word starts as the list of its characters (Unicode code points)
followed by one extra symbol, the end-of-word marker `"</w>"`. The word `low` is
`l o w </w>`: four symbols. A symbol is a string; merging the pair `(left, right)`
creates the symbol `left + right`.

**One step.** Repeat the following `n_merges` times.

1. **Count.** For every word and every position `i` in its current symbol list, the
   pair `(symbols[i], symbols[i + 1])` receives the word's frequency. Every adjacent
   position counts, also overlapping ones: in a word with the symbols `a a a </w>`
   and frequency 3, the pair `(a, a)` receives $2 \cdot 3 = 6$.
2. **Stop early** if no pair exists (every word is a single symbol, or the corpus is
   empty): return the merges learned so far. A pair with count 1 is still merged.
3. **Select** the pair with the highest count. Among pairs with the same count, take
   the smallest in Python's tuple order: compare the left strings, then the right
   strings (strings compare by code point, so `"</w>"` sorts before every letter and
   `"es"` before `"t"`). The result does not depend on the order of the dict.
4. **Record** the pair as the next merge.
5. **Apply** it to every word: scan the symbol list from left to right, and wherever
   `left` is immediately followed by `right`, replace the two symbols with
   `left + right` and continue after them. Occurrences therefore never overlap, and
   in a run the leftmost wins: with the merge `(a, a)`, `a a a </w>` becomes
   `aa a </w>`. Only whole symbols match: the merge `(a, b)` does not touch the
   symbols `xa b`.

The function does not modify `corpus_words`.

Write the procedure yourself: do not use a tokenizer library.

## Example

The corpus of the original paper:

```python
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
learn_merges(corpus, 10)
# [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w"),
#  ("e", "w"), ("ew", "est</w>"), ("n", "ewest</w>"), ("low", "</w>"), ("d", "est</w>")]
```

| step | highest count | pairs with that count | merge |
|---|---|---|---|
| 1 | 9 | `(e, s)`, `(s, t)`, `(t, </w>)` | `(e, s)` |
| 2 | 9 | `(es, t)`, `(t, </w>)` | `(es, t)` |
| 3 | 9 | `(est, </w>)` | `(est, </w>)` |
| 4 | 7 | `(l, o)`, `(o, w)` | `(l, o)` |
| 5 | 7 | `(lo, w)` | `(lo, w)` |
| 6 | 6 | `(e, w)`, `(n, e)`, `(w, est</w>)` | `(e, w)` |

In step 1, `e s` occurs in `newest` (6) and `widest` (3). In step 6 the words are
`low </w>`, `low e r </w>`, `n e w est</w>`, and `w i d est</w>`; three pairs occur 6
times, and `("e", "w")` is the smallest.

Counting and applying treat a run differently:

```python
learn_merges({"aaa": 3, "ab": 5}, 2)  # [("a", "a"), ("a", "b")]
```

`(a, a)` is counted at both positions of `a a a </w>`, $2 \cdot 3 = 6$, which beats
`(a, b)` and `(b, </w>)` with 5 each. Applying it gives `aa a </w>`, one merged symbol
per word, not two.

The procedure stops when nothing is left to merge:

```python
learn_merges({"ba": 4}, 4)  # [("a", "</w>"), ("b", "a</w>")]
```

There are two initial pairs, `(b, a)` and `(a, </w>)`, each with count 4, and
`("a", "</w>")` is the smaller. After the second merge the word is the single
symbol `ba</w>`.

## Constraints

- At most 500 distinct words, each of 1 to 30 characters, with frequencies from 1 to
  $10^6$. No word contains whitespace or the character `<`.
- `0 <= n_merges <= 300`.
- The merge list is compared exactly, in order. Each element must be a tuple (or
  list) of two strings.
- There is no speed requirement beyond the time limit. Recounting all pairs from
  scratch in every step is fast enough at this size.

## Why this matters

Every BPE tokenizer you load for inference is the output of this loop, run once over
a large corpus, and the merge list it produces is the rank order that the encoder of
`bpe-encode` replays. You need the procedure itself when you extend a vocabulary for a
new domain, build a tokenizer for a draft model, or debug why two "identical"
tokenizers disagree. The usual cause is one of the choices fixed above: implementations
differ in how they break ties, whether the end-of-word marker is its own symbol, and
when they stop, and each choice yields a different table from the same corpus. Since
the model's embedding rows are indexed by that table, a tokenizer rebuilt with a
different tie rule is a different tokenizer.
