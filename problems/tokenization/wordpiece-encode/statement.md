Implement BERT's WordPiece encoder. Given a single word, split it from left to right into the longest matching pieces in a vocabulary, adding the `##` prefix to every piece after the first. The scan is greedy with no backtracking, and any word that cannot be fully split becomes a single unknown token.

![This problem is the Encode step, splitting one word greedily into vocabulary pieces.](assets/map.svg)

## Specification

```python
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]: ...
```

- `word`: one word, already normalized and split from the surrounding text (no whitespace). It may be empty and may contain any Unicode characters.
- `vocab`: dictionary mapping token strings to integer IDs. Word-initial tokens are stored as plain strings (for example, `"un"`), while continuation tokens carry the `##` prefix (for example, `"##able"`). These are distinct entries with distinct IDs.
- `unk`: the unknown token string (for example, `"[UNK]"`). It is always present in `vocab`.
- `max_chars`: maximum word length, in characters, that the encoder will attempt to split.

A **character** is one Unicode code point (one element of a Python `str`), so `len(word)` is the word's length in characters.

Compute the list of token IDs as follows:

1. If `word` is empty, return `[]`.
2. If `len(word) > max_chars`, return `[vocab[unk]]`. A word of exactly `max_chars` characters is split normally.
3. Otherwise, set `start = 0` and repeat until `start == len(word)`:
   - Consider substrings `word[start:end]` for `end` from `len(word)` down to `start + 1`, **longest first**. The candidate token is `word[start:end]` when `start == 0`, and `"##" + word[start:end]` when `start > 0`.
   - Take the first candidate that exists in `vocab`. Append its ID to the result and set `start = end`.
   - If no candidate is in `vocab`, stop immediately and return `[vocab[unk]]`. Discard any pieces matched so far so the **whole word** becomes one unknown token.

There is no backtracking. Once a piece is chosen, a later failure never causes the encoder to try a shorter earlier piece.

The `##` prefix is only added when forming continuation candidates. Any `#` characters already inside `word` are left untouched. Return a new list of Python `int` values without modifying `vocab`.

Implement the algorithm yourself without using a tokenizer library.

## Example

```python
vocab = {
    "[UNK]": 0,
    "un": 1,
    "##aff": 2,
    "##able": 3,
    "aff": 4,
    "able": 5,
    "##a": 6,
    "u": 7,
    "##n": 8,
    "##ly": 9,
    "abc": 10,
    "ab": 11,
    "##cd": 12,
}

wordpiece("unaffable", vocab, "[UNK]", 100)  # [1, 2, 3]   un ##aff ##able
wordpiece("able", vocab, "[UNK]", 100)  # [5]         able, not ##able
wordpiece("unable", vocab, "[UNK]", 100)  # [1, 3]      un ##able
wordpiece("unaffably", vocab, "[UNK]", 100)  # [0]
wordpiece("abcd", vocab, "[UNK]", 100)  # [0]
wordpiece("unaffable", vocab, "[UNK]", 8)  # [0]         9 characters > 8
```

- For `"unaffable"`, at `start = 0` the candidates are `"unaffable"`, `"unaffabl"`, and so on down to `"un"`, which is the first match in `vocab` (`"u"` is also present, but `"un"` is longer). At `start = 2`, the encoder tries `"##affable"`, `"##affabl"`, `"##affab"`, `"##affa"`, and matches `"##aff"`. At `start = 5`, `"##able"` matches immediately.
- For `"unaffably"`, the encoder matches `"un"` and `"##aff"` as before. At `start = 5`, the longest match is `"##a"`. At `start = 6`, none of `"##bly"`, `"##bl"`, or `"##b"` is in `vocab`, so the whole word becomes `[0]` rather than `[1, 2, 6, 0]`.
- For `"abcd"`, the longest first piece is `"abc"`, leaving `"##d"`, which is missing. The split `"ab"` + `"##cd"` exists in `vocab`, but the encoder does not backtrack and returns `[0]`.

## Constraints

- `len(word) <= 400`, `1 <= max_chars <= 200`, and `vocab` has at most 30,000 entries.
- Token IDs are compared for exact equality.
- Trying every `end` for every `start` requires at most `max_chars * (max_chars + 1) / 2` dictionary lookups, which fits comfortably within the time limit.

## Why this matters

Decoder LLMs moved to BPE and SentencePiece, but BERT-family embedding models, cross-encoder rerankers, and many text classifiers still use WordPiece. In a retrieval stack, these encoders often tokenize far more text than the downstream LLM ever sees.

The exact token IDs are part of the model's input contract. A port that emits `[un, ##aff, ##a, [UNK]]` instead of `[UNK]`, or backtracks to find a valid split, produces token sequences the model was never trained on. Nothing crashes, and the only symptom is lower recall on rare words, names, and identifiers.
