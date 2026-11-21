Implement the WordPiece encoder of BERT, which is still the tokenizer of most embedding
and reranking models: split one word into the longest vocabulary pieces, left to
right, marking every piece after the first with the continuation prefix `##`. The
algorithm is greedy and never reconsiders, and a word that cannot be finished becomes
a single unknown token as a whole.

![This problem is the Encode step, splitting one word greedily into vocabulary pieces.](assets/map.svg)

## Specification

```python
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]: ...
```

- `word`: one word, already normalized and split off from the text (no whitespace).
  It may be empty and may contain any Unicode characters.
- `vocab`: the vocabulary, mapping token strings to ids. Tokens that start a word are
  stored as they are (`"un"`); tokens that continue a word are stored with the prefix
  `##` (`"##able"`). The two are different entries with different ids.
- `unk`: the unknown token, for example `"[UNK]"`. It is always a key of `vocab`.
- `max_chars`: the longest word, in characters, that is split at all.

A **character** is one Unicode code point, that is, one element of a Python `str`;
`len(word)` is the word's length in characters.

The result is the list of ids computed as follows.

1. If `word` is empty, return `[]`.
2. If `len(word) > max_chars`, return `[vocab[unk]]`. A word of exactly `max_chars`
   characters is split normally.
3. Otherwise set `start = 0` and repeat until `start == len(word)`:
   - Consider the substrings `word[start:end]` for `end = len(word)` down to
     `end = start + 1`, **longest first**. The candidate token is the substring itself
     when `start == 0`, and `"##"` followed by the substring when `start > 0`.
   - Take the first candidate that is a key of `vocab`: append its id to the result
     and continue with `start = end`.
   - If no candidate is in `vocab`, stop and return `[vocab[unk]]`: the **whole word**
     becomes one unknown token, and the pieces found so far are discarded.

There is no backtracking: once a piece is taken, a later failure does not make the
algorithm try a shorter piece in its place.

The prefix `##` is only ever *added* to candidates; nothing is done about a `word`
that happens to contain `#` characters itself. The function returns a new list of
Python ints and does not modify `vocab`.

Write the algorithm yourself: do not use a tokenizer library.

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

`unaffable`: at `start = 0` the candidates are `unaffable`, `unaffabl`, ..., down to
`un`, the first one in the vocabulary (`u` is there too, but `un` is longer). At
`start = 2` the candidates are `##affable`, `##affabl`, `##affab`, `##affa`, then
`##aff`. At `start = 5`, `##able` matches at once.

`unaffably`: `un` and `##aff` are found as before. At `start = 5` the longest match is
`##a`, and at `start = 6` none of `##bly`, `##bl`, `##b` is in the vocabulary. The
result is `[0]`, not `[1, 2, 6, 0]`.

`abcd`: the longest first piece is `abc`, and then `##d` is missing. The split
`ab` + `##cd` exists, but the algorithm does not go back for it.

## Constraints

- `len(word) <= 400`, `1 <= max_chars <= 200`, and `vocab` has at most 30,000 entries.
- Ids are compared exactly.
- There is no speed requirement beyond the time limit: trying every `end` for every
  `start` is at most `max_chars * (max_chars + 1) / 2` dictionary lookups.

## Why this matters

Decoder LLMs moved to BPE and SentencePiece, but the encoders that sit next to them in
a serving stack did not: BERT-family embedding models, cross-encoder rerankers, and
many classifiers are trained on WordPiece ids, and a retrieval pipeline tokenizes far
more text with WordPiece than its LLM ever reads. The ids are the model's input
contract. A port that emits `[un, ##aff, ##a, [UNK]]` where the reference emits
`[UNK]`, or that finds a "better" split by backtracking, produces sequences of
plausible length that the model was never trained on, and the only symptom is that
recall drops on text with rare words, names, and identifiers.
