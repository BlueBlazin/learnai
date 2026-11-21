Implement the encoder and decoder of a byte-level BPE tokenizer, the kind GPT-style
models use: text becomes UTF-8 bytes, and a ranked list of merges turns runs of bytes
into token ids. The difficulty is that the result is defined by the *order* in which
merges are applied, so a tokenizer that is nearly right produces ids the model was
never trained on.

![This problem is the Encode step, where ranked merges turn the bytes of the text into token ids.](assets/map.svg)

## Specification

```python
def encode(text: str, merges: list[tuple[int, int]]) -> list[int]: ...
def decode(ids: list[int], merges: list[tuple[int, int]]) -> str: ...
```

**Token ids.** Ids `0` to `255` are the 256 possible byte values. The merge table
`merges` defines the rest: `merges[i] = (left, right)` says that the token `left`
immediately followed by the token `right` merges into the token with id `256 + i`.
The position `i` is also the merge's **rank**, and a lower rank means a higher
priority. The bytes of token `256 + i` are the bytes of `left` followed by the bytes of
`right`.

**`encode(text, merges)`** returns the list of token ids computed as follows.

1. Start with the UTF-8 encoding of `text`, one id per byte.
2. Among all pairs of adjacent ids in the current sequence, find the pairs that appear
   in `merges`. If there are none, stop: the current sequence is the result.
3. Take the one with the lowest rank. Position in the text plays no role in this
   choice, and neither does how often the pair occurs.
4. Replace the occurrences of that pair with its merged id `256 + rank` in a single
   pass from left to right: when the pair is found at positions `j, j + 1`, emit the
   merged id and continue at position `j + 2`. Occurrences therefore never overlap,
   and in a run the leftmost ones win: with the merge `(a, a)`, the run `a a a`
   becomes `aa a`, and `a a a a` becomes `aa aa`.
5. Go back to step 2 with the new sequence.

The pairs in `merges` are distinct, so two different merges never tie on rank, and all
occurrences of the chosen pair are handled by step 4.

**`decode(ids, merges)`** concatenates the bytes of every token in `ids`, in order,
and decodes the result as UTF-8. Invalid byte sequences are replaced with U+FFFD, as
`bytes.decode("utf-8", errors="replace")` does. The whole byte string is decoded at
once: the bytes of one character may be spread over several tokens.

Details that the functions must handle:

- `text` may be empty (the result is `[]`) and may contain any Unicode characters,
  including ones that take two, three, or four bytes in UTF-8.
- `merges` may be empty; then `encode` returns the bytes.
- `left` and `right` may be any id in `range(256 + len(merges))`. A table produced by
  BPE training only refers to earlier merges (`left, right < 256 + i`), but that is
  **not guaranteed here**: a merge may refer to a token that a *later* merge creates.
  The steps above apply unchanged. No token contains itself, directly or through other
  merges, so every token has a finite byte string.
- For `decode`, every id is in `range(256 + len(merges))`; `ids` may be empty and need
  not be the output of `encode`.
- Both functions return new objects and do not modify their arguments. `encode`
  returns a list of Python ints.

Write the algorithm yourself: do not use a tokenizer library.

## Example

Let `a = 97`, `b = 98`, `c = 99`, `d = 100`.

```python
merges = [(97, 97), (97, 98), (256, 257)]      # 256 = "aa", 257 = "ab", 258 = "aaab"
encode("aaabdaaabac", merges)                  # [258, 100, 258, 97, 99]
decode([258, 100, 258, 97, 99], merges)        # "aaabdaaabac"
```

Step by step, writing merged tokens in brackets:

| sequence | pairs present that are merges | applied |
|---|---|---|
| `a a a b d a a a b a c` | `(a, a)` rank 0, `(a, b)` rank 1 | rank 0 |
| `[aa] a b d [aa] a b a c` | `(a, b)` rank 1 | rank 1 |
| `[aa] [ab] d [aa] [ab] a c` | `([aa], [ab])` rank 2 | rank 2 |
| `[aaab] d [aaab] a c` | none | stop |

![The eleven bytes of the example after each applied merge, with the id of every token under it.](assets/merge-steps.svg)

In the first step each run `a a a` becomes `[aa] a`, not `a [aa]`: the scan goes from
left to right. Note also that `(a, b)` was applicable from the start and had to wait
for rank 0.

Rank decides, not position:

```python
encode("abc", [(98, 99), (97, 98)])            # [97, 256]   (b, c) has rank 0
encode("abc", [(97, 98), (98, 99)])            # [256, 99]   (a, b) has rank 0
```

A merge may depend on a later one. Here token 256 is `(257, c)` and token 257 is
`(a, b)`; rank 1 is the only applicable merge at first, and it makes rank 0 applicable:

```python
encode("abc", [(257, 99), (97, 98)])           # [256]
```

Encoding starts from bytes, not characters. `"é"` is the two bytes `195, 169`:

```python
encode("é", [])                                # [195, 169]
encode("héé", [(195, 169), (104, 256)])        # [257, 256]
decode([104, 195], [])                         # "h�"   a truncated character
```

## Constraints

- `len(text) <= 2000` characters and `len(merges) <= 1000`.
- Ids are compared exactly, and so are decoded strings.
- There is no speed requirement beyond the time limit. The direct algorithm, one scan
  of the sequence per applied merge, is fast enough.
- `decode(encode(s, merges), merges) == s` must hold for every string `s` and every
  valid table; the tests check it on random Unicode strings.

## Why this matters

The tokenizer is the first stage of every inference request, and the model only works
on the exact ids it saw in training. An encoder that applies merges in a slightly
different order still returns ids that decode to the right text, so nothing crashes:
the model just sees an unfamiliar segmentation and quietly gets worse. This is a
classic source of quality gaps between a reference implementation and a port to
another runtime. On the way out, the decoder runs once per generated token in a
streaming server, where multi-byte characters routinely arrive split across tokens,
and a decoder that converts each token on its own emits replacement characters in the
middle of words.
