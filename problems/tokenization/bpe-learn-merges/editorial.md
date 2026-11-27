## The idea

Byte Pair Encoding for tokenization (Sennrich, Haddow, and Birch, 2016) starts from individual characters and grows a vocabulary greedily. At each step, the two adjacent symbols that appear together most often are merged into one symbol. After $n$ merges, the vocabulary contains the base characters plus $n$ new symbols. Frequent words collapse into single symbols, while rare words stay split into shorter subwords.

Operating on a dictionary of word frequencies instead of raw running text is the first optimization from the paper. Because each distinct word is stored once, its frequency must weight its pair counts. A word with frequency 6 adds 6 to each of its adjacent pairs. Appending `</w>` makes word boundaries explicit so that a suffix like `est</w>` and an internal substring like `est` remain distinct tokens, and so the original text can be reconstructed cleanly.

Three conventions make the algorithm deterministic:

- **Counting pairs.** Every adjacent position counts, matching `get_stats` in the original paper (which iterates over `range(len(symbols) - 1)`) and `subword-nmt`. In `a a a`, the pair `(a, a)` counts twice even though only one merge can apply there. The count measures how often the pair is adjacent, not how many replacements it will perform.
- **Breaking ties.** The paper's code uses `max(pairs, key=pairs.get)`, which returns whichever tied pair was inserted into the dictionary first and therefore depends on corpus order. Production implementations fix a deterministic tie-breaker. This problem picks the smallest pair in tuple order.
- **Applying merges.** Scan left to right without overlap, matching whole symbols only.

Pair counts must be recomputed after every merge. Merging `(e, s)` destroys `(w, e)` and `(s, t)` at those positions while creating `(w, es)` and `(es, t)`. The next most frequent pair is often one that did not exist in the previous step.

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

`_merge` is the same helper used in `bpe-encode`, operating on strings instead of integer ids.

## Common mistakes

- **Breaking ties by first occurrence or largest pair.** Using `max(counts, key=counts.get)` makes the output depend on dictionary insertion order. Breaking ties toward the largest pair (`max(stats, key=lambda x: (stats[x], x))`) is the rule used by `subword-nmt`, which differs from the smallest-pair rule specified here.
- **Unweighted counts.** Counting each distinct word once learns the most frequent pairs in the unique word list rather than in the corpus.
- **Counting only non-overlapping pairs.** Skipping overlapping occurrences in runs like `aaa` seems reasonable because only one merge can apply, but it diverges from the standard BPE counting rule and produces a different merge table.
- **Using stale counts.** Taking the top $n$ pairs from the initial character counts without recounting only ever merges single characters.
- **Using string replacement to apply a merge.** `" ".join(symbols).replace("a b", "ab")` mistakenly matches the end of `xa` followed by `b`. `subword-nmt` wraps its regex with `(?<!\S)` and `(?!\S)` to avoid this bug.
- **Merging only the first occurrence in a word.** Stopping after one replacement per word leaves inputs like `ba ba` half merged.
- **Omitting `</w>` or attaching it to the last character.** Gluing `</w>` onto the final character is a valid alternative (used by `subword-nmt`), but it changes the merge sequence—for example, producing `(s, t</w>)` instead of `(s, t)` followed by `(st, </w>)`.
- **Stopping at count 1.** `subword-nmt` stops when the best pair appears fewer than `--min-frequency` times (default 2). This specification stops only when no pairs remain.

## Complexity

Let $L$ be the total number of symbols across all distinct words and $n$ be the number of merges. Each step recounts pairs in $O(L)$ time and rewrites the symbol lists in $O(L)$ time, giving $O(n \cdot L)$ total time. At 500 words of up to 30 characters and 300 merges, that is only a few million operations.

Production trainers handle millions of distinct words and tens of thousands of merges by updating counts incrementally. They maintain an index from each pair to the words containing it and a priority queue of pair counts, updating only the affected words on each merge. Because the tie-breaking rule is fixed, incremental training produces the exact same merge table.

## In the wild

GPT-2-style byte-level BPE runs this same loop over bytes instead of characters and marks word starts with a leading space rather than marking word ends. The resulting `merges.txt` file or `merges` array in `tokenizer.json` stores the exact list returned here, one pair per line in learned order. SentencePiece's BPE mode and the Hugging Face `tokenizers` trainer implement the same loop with their own tie-breaking and frequency-threshold conventions. Training runs once before a model is trained, and inference only replays the saved merge table.
