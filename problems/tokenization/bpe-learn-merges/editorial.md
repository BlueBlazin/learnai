## The idea

Byte pair encoding as a tokenizer (Sennrich, Haddow, and Birch, 2016) starts from
characters and grows a vocabulary greedily: whichever two adjacent symbols occur
together most often become one symbol. After $n$ merges the vocabulary holds the
characters plus $n$ new symbols, frequent words have collapsed into single symbols,
and rare words remain sequences of shorter pieces.

Working on a dictionary of word frequencies instead of the running text is the first
optimization of the paper, and it is why frequencies must weight the counts: a word
that occurs 6 times contributes each of its pairs 6 times. The end-of-word marker
`</w>` makes word boundaries part of the symbols, so that `est</w>` (a suffix) and
`est` (inside a word) can be learned as different tokens, and so that the original
text can be restored from the tokens.

The procedure is deterministic only if three things are fixed:

- **How pairs are counted.** Every adjacent position counts, as in the paper's
  `get_stats` (it loops over `range(len(symbols) - 1)`) and in subword-nmt. In a run
  `a a a` the pair `(a, a)` therefore counts twice, although a merge can only be
  applied once there. The count is "how often is this pair adjacent", not "how many
  merges would this produce".
- **Ties.** The paper's code takes `max(pairs, key=pairs.get)`, which returns
  whichever maximal pair was inserted into the dict first, so its result depends on
  the order of the corpus. Real implementations pick a rule; this problem uses the
  smallest pair.
- **How a merge is applied.** Left to right, without overlap, on whole symbols.

Counts must be taken again after every merge. Merging `(e, s)` removes the pairs
`(w, e)` and `(s, t)` at those positions and creates `(w, es)` and `(es, t)`; the next
most frequent pair is usually one that did not exist before.

## Reference solution (Python)

```python
END = "</w>"


def learn_merges(corpus_words, n_merges):
    words = [([*word, END], freq) for word, freq in corpus_words.items()]
    merges = []
    for _ in range(n_merges):
        counts = Counter()
        for symbols, freq in words:
            for pair in pairwise(symbols):
                counts[pair] += freq
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        words = [(_merge(symbols, best), freq) for symbols, freq in words]
    return merges


def _merge(symbols, pair):
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
```

`_merge` is the function of the same name in `bpe-encode`, on strings instead of ids.

## Common mistakes

- **Ties by first occurrence** (`max(counts, key=counts.get)`): the table changes when
  the corpus dict is built in a different order. **Ties toward the largest pair** is
  the rule of subword-nmt (`max(stats, key=lambda x: (stats[x], x))`), correct there
  and wrong here: always check which rule a table was built with.
- **Unweighted counts.** Counting each distinct word once learns the pairs of the
  *vocabulary*, not of the text.
- **Counting only non-overlapping occurrences.** Plausible reasoning ("`aaa` can only
  be merged once") and a different table from every reference implementation.
- **Stale counts.** Taking the top $n$ pairs of the initial character counts returns
  only pairs of single characters.
- **String replacement to apply a merge.** `" ".join(symbols).replace("a b", "ab")`
  also matches the end of the symbol `xa` followed by `b`. subword-nmt guards its
  regular expression with `(?<!\S)` and `(?!\S)` for exactly this reason.
- **Merging only the first occurrence in a word**, which leaves `ba ba` half merged.
- **No end-of-word symbol, or the marker glued to the last character.** The second is
  a legitimate representation (subword-nmt uses it) that yields different merges:
  `(s, t</w>)` instead of `(s, t)` then `(st, </w>)`.
- **Stopping at count 1.** subword-nmt stops when the best pair occurs fewer than
  `--min-frequency` times (2 by default). The statement here stops only when no pair
  is left.

## Complexity

With $W$ distinct words of total length $L$ symbols, each step recounts in $O(L)$ and
rewrites in $O(L)$: $O(n \cdot L)$ for $n$ merges. At 500 words of up to 30 characters
and 300 merges that is a few million operations. Real trainers, with millions of
distinct words and tens of thousands of merges, update the counts incrementally: only
the words that contain the merged pair change, an index from pair to words finds them,
and a priority queue yields the next best pair. The output is identical; that is the
point of pinning down the tie rule.

## In the wild

GPT-2-style byte-level BPE runs this loop over bytes instead of characters and marks
word starts (a leading space) instead of word ends, and the resulting `merges.txt` or
the `merges` array in `tokenizer.json` is exactly the list returned here, one pair per
line in learned order. SentencePiece's BPE mode and the Hugging Face `tokenizers`
trainer are other implementations of the same loop with their own conventions for
ties and limits. This is tokenizer construction, done once, before any model exists;
at inference time only the finished table is used.
