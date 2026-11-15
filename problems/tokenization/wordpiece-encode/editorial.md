## The idea

A WordPiece vocabulary holds two kinds of entries: pieces that can start a word, and
pieces that can continue one, stored with a `##` prefix. Encoding a word is a greedy
scan (Devlin et al., 2018, following Wu et al., 2016): at the current position take
the longest piece of the right kind, move past it, repeat. BERT's reference
implementation (`WordpieceTokenizer.tokenize` in `google-research/bert`,
`tokenization.py`) is exactly the loop below, including its two blunt rules: a word
longer than `max_input_chars_per_word` (200 by default there) is not examined at all,
and a word in which some position has no matching piece becomes the single token
`[UNK]`.

Greedy is not optimal, and it is not meant to be. It does not find the split with the
fewest pieces, and it gives up on words that a search could split. What matters is
that it is the function the model was trained behind: the model has seen `[UNK]` for
`abcd` and has never seen `ab ##cd`.

Why longest-first works as a loop: for a fixed `start`, the candidates are nested
prefixes of the rest of the word, so "the longest one in the vocabulary" is found by
trying `end` from the right and stopping at the first hit. There is at most one hit
to take per position and no state besides `start`.

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

- **Shortest match.** Iterating `end` upward returns the first single character that
  is in the vocabulary. Every word still splits, into the longest possible sequence.
- **The prefix on the wrong pieces.** Looking up the first piece with `##`, or later
  pieces without it, confuses the two kinds of entries. With a vocabulary that has
  both `able` and `##able`, the ids differ and nothing fails.
- **`[UNK]` for the failing part only.** Emitting the pieces found so far and then one
  unknown token looks more informative and is not what BERT does: the whole word is
  replaced.
- **Backtracking.** Trying a shorter earlier piece when a later position fails turns
  some `[UNK]` words into splits the model has never seen.
- **The length limit.** The comparison is `>` and the unit is characters. Counting
  UTF-8 bytes makes a 70-character Japanese word "too long" for a limit of 200.
- **An empty word** returns `[]`, not `[UNK]`: the scan never starts.

## Complexity

For a word of $n \le$ `max_chars` characters the scan performs at most $n(n+1)/2$
lookups, each hashing a substring of up to $n$ characters: $O(n^3)$ character
operations in the worst case. The length limit is what keeps that bounded, which is
why over-long words are rejected before the scan and not after it.
Linear-time WordPiece exists (Song et al., 2021, "Fast WordPiece Tokenization", a
trie with failure links) and gives the same output.

## In the wild

The full BERT tokenizer runs this after a normalization stage (cleanup, optional
lowercasing and accent stripping, splitting on whitespace and punctuation), so the
`word` here is one of that stage's outputs. Then it adds `[CLS]` and `[SEP]`.
Sentence-embedding and reranking models built on BERT, MiniLM, and their successors
ship a `vocab.txt` whose line numbers are the ids used here. The Hugging Face
tokenizers expose the same three parameters as `unk_token`,
`continuing_subword_prefix`, and `max_input_chars_per_word`.
