Implement the procedure that learns a Byte Pair Encoding (BPE) merge table from a word-frequency corpus. Repeatedly find the most frequent adjacent pair of symbols and merge it into a single new symbol. Because the resulting merge table must be reproducible, the counting rule, tie-breaking rule, and merge application order are all specified below.

## Specification

```python
def learn_merges(corpus_words: dict[str, int], n_merges: int) -> list[tuple[str, str]]: ...
```

- `corpus_words`: the corpus as word frequencies. Each key is a word, and each value is its occurrence count (a positive `int`). The corpus is already split into words, and merges never cross word boundaries.
- `n_merges`: the number of merges to learn (a non-negative `int`).
- Returns the learned merges in order as a list of `(left, right)` string tuples.

**Symbols.** Every word starts as a list of its characters (Unicode code points) followed by the end-of-word marker `"</w>"`. For example, `"low"` starts as the four symbols `l`, `o`, `w`, `</w>`. Each symbol is a string, and merging `(left, right)` creates the new symbol `left + right`.

**One step.** Repeat the following `n_merges` times.

1. **Count.** For every word and every adjacent position `i` in its current symbol list `symbols`, add the word's frequency to the count of `(symbols[i], symbols[i + 1])`. Every adjacent position counts, including overlapping ones. For example, in a word with symbols `a a a </w>` and frequency 3, the pair `(a, a)` appears twice and receives $2 \cdot 3 = 6$.
2. **Stop early.** If no adjacent pair exists (every word is a single symbol, or the corpus is empty), return the merges learned so far. A pair with count 1 is still merged.
3. **Select.** Pick the pair with the highest count. Break ties by choosing the smallest pair in Python's tuple order. Tuple comparison compares the left strings first and then the right strings by Unicode code point, so `"</w>"` sorts before every letter and `"es"` sorts before `"t"`. The result must not depend on the iteration order of `corpus_words`.
4. **Record.** Append the chosen pair to the merge list.
5. **Apply.** Scan each word's symbol list from left to right. Whenever `left` is immediately followed by `right`, replace both symbols with `left + right` and continue after them. Replacements never overlap, and in a run of repeated symbols the leftmost match wins. For example, merging `(a, a)` turns `a a a </w>` into `aa a </w>`. Only whole symbols match, so merging `(a, b)` does not modify `xa b`.

Do not modify `corpus_words`. Implement the procedure yourself without using a tokenizer library.

## Example

On the corpus from the original BPE paper:

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

In step 1, `e s` appears in `newest` (6) and `widest` (3) for a total of 9. By step 6, the words are `low </w>`, `low e r </w>`, `n e w est</w>`, and `w i d est</w>`. Three pairs tie with count 6, and `("e", "w")` is the smallest.

Counting and applying handle runs differently:

```python
learn_merges({"aaa": 3, "ab": 5}, 2)  # [("a", "a"), ("a", "b")]
```

The pair `(a, a)` is counted at both adjacent positions in `a a a </w>`, giving $2 \cdot 3 = 6$. That beats `(a, b)` and `(b, </w>)`, which each have count 5. Applying `(a, a)` produces `aa a </w>`, creating one merged symbol per occurrence of the word rather than two.

The loop stops early when no pairs remain:

```python
learn_merges({"ba": 4}, 4)  # [("a", "</w>"), ("b", "a</w>")]
```

The two initial pairs `(b, a)` and `(a, </w>)` tie at count 4, and `("a", "</w>")` is smaller. After the second merge, the word is the single symbol `ba</w>`, so no pairs remain.

## Constraints

- At most 500 distinct words, each of 1 to 30 characters, with frequencies from 1 to $10^6$. No word contains whitespace or the character `<`.
- `0 <= n_merges <= 300`.
- The merge list is compared exactly, in order. Each element must be a tuple (or list) of two strings.
- There is no speed requirement beyond the time limit. Recounting all pairs from scratch in every step is fast enough at this scale.

## Why this matters

Every BPE tokenizer used at inference time comes from running this loop once over a large corpus. The resulting merge list defines the rank order that the encoder in `bpe-encode` replays.

You need this procedure when extending a vocabulary for a new domain, building a tokenizer for a draft model, or debugging why two supposedly identical tokenizers disagree. Those disagreements almost always come from the conventions pinned down above: tie-breaking order, whether the end-of-word marker is a separate symbol, and the stopping condition. Because a model's embedding rows are indexed by the tokenizer's table, changing any of these rules produces a different tokenizer.
