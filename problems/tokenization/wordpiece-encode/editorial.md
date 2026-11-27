## The idea

A WordPiece vocabulary holds two kinds of entries. Pieces that can start a word are stored as plain strings, and pieces that can continue a word are stored with a `##` prefix. Encoding a word is a greedy left-to-right scan (Devlin et al., 2018, following Wu et al., 2016). At the current position, take the longest matching piece of the right kind, advance past it, and repeat. BERT's reference implementation (`WordpieceTokenizer.tokenize` in `google-research/bert`, `tokenization.py`) uses this exact loop along with two blunt rules. First, any word longer than `max_input_chars_per_word` (200 by default) is rejected before scanning. Second, if any position has no matching piece, the entire word collapses to `[UNK]`.

Greedy matching is not optimal, and it is not meant to be. It does not guarantee the fewest pieces, and it fails on words that a backtracking search could split. What matters is matching the exact function the model was trained on. During training, the model saw `[UNK]` for `"abcd"` and never saw `"ab"` followed by `"##cd"`.

For a fixed `start`, the candidates are nested prefixes of `word[start:]`. Scanning `end` downward from `len(word)` to `start + 1` and stopping at the first dictionary hit finds the longest match directly. Each step takes one piece and requires no state beyond `start`.

## Reference solution (Python)

```python
def wordpiece(word, vocab, unk, max_chars):
    if len(word) > max_chars:
        return [vocab[unk]]
    ids = []
    start = 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                start = end
                break
        else:
            return [vocab[unk]]
    return ids
```

## Common mistakes

- **Shortest match.** Iterating `end` upward returns the first single character in `vocab`, splitting every word into the longest possible sequence of pieces.
- **Prefix on the wrong pieces.** Adding `##` to the first piece, or omitting it on later pieces, mixes up the two entry types. When `vocab` contains both `"able"` and `"##able"`, the encoder still succeeds but emits the wrong IDs.
- **Emitting `[UNK]` only for the failing suffix.** Returning the pieces matched so far followed by `[UNK]` looks more informative, but BERT replaces the whole word with a single `[UNK]`.
- **Backtracking.** Retrying a shorter earlier piece when a later position fails turns `[UNK]` words into subword sequences the model was never trained on.
- **Wrong length check or unit.** The limit check must use `>` (not `>=`) and count Unicode characters rather than UTF-8 bytes. Counting bytes causes a 70-character Japanese word to exceed a 200-character limit.
- **Empty word.** An empty string returns `[]`, not `[vocab[unk]]`, because the scan never runs.

## Complexity

For a word of $n \le \texttt{max\_chars}$ characters, the scan performs at most $n(n + 1)/2$ dictionary lookups. Each lookup slices and hashes a substring of up to $n$ characters, giving $O(n^3)$ character operations in the worst case. Checking `len(word) > max_chars` before the loop keeps this work bounded. Linear-time WordPiece algorithms exist (such as Song et al., 2021, "Fast WordPiece Tokenization", using a trie with failure links) and produce the exact same output.

## In the wild

The full BERT tokenizer runs a normalization stage before WordPiece (text cleanup, optional lowercasing and accent stripping, and splitting on whitespace and punctuation), then wraps the result in `[CLS]` and `[SEP]`. Sentence-embedding and reranking models based on BERT, MiniLM, and their successors ship a `vocab.txt` whose line numbers are the IDs used here. Hugging Face tokenizers expose the same three parameters as `unk_token`, `continuing_subword_prefix`, and `max_input_chars_per_word`.
