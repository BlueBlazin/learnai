Implement the encoder and decoder of a byte-level Byte Pair Encoding (BPE) tokenizer. The encoder converts text into UTF-8 bytes and repeatedly merges adjacent token pairs according to a ranked merge table. The decoder expands token ids back into bytes and decodes them as UTF-8.

![This problem is the Encode step, where ranked merges turn the bytes of the text into token ids.](assets/map.svg)

## Specification

```python
def encode(text: str, merges: list[tuple[int, int]]) -> list[int]: ...
def decode(ids: list[int], merges: list[tuple[int, int]]) -> str: ...
```

**Token ids.** Ids `0` through `255` represent the 256 possible byte values. The merge table `merges` defines all higher ids. Entry `merges[i] = (left, right)` specifies that token `left` immediately followed by token `right` merges into token id `256 + i`. The index `i` is the merge's **rank**, and a lower rank means higher priority. The bytes of token `256 + i` are the bytes of `left` followed by the bytes of `right`.

**`encode(text, merges)`** computes the token id list as follows:

1. Convert `text` to its UTF-8 bytes, producing one initial token id per byte.
2. Find all adjacent token pairs in the current sequence that appear in `merges`. If none appear, stop and return the current sequence.
3. Choose the matching pair with the lowest rank. Neither position in the sequence nor frequency affects this choice.
4. Replace occurrences of that pair with `256 + rank` in a single left-to-right pass. When the pair appears at positions `j` and `j + 1`, emit the merged id and advance to `j + 2`. Matches never overlap, and leftmost matches win in runs. For example, with merge `(a, a)`, the run `a a a` becomes `aa a`, and `a a a a` becomes `aa aa`.
5. Repeat from step 2 with the updated sequence.

All pairs in `merges` are distinct, so two different merges never tie on rank.

**`decode(ids, merges)`** concatenates the bytes of every token in `ids` in order and decodes the full byte string as UTF-8. Invalid byte sequences are replaced with U+FFFD, matching `bytes.decode("utf-8", errors="replace")`. Decode the entire byte sequence at once rather than token by token, because one character's bytes may span multiple tokens.

Edge cases and rules:

- `text` may be empty (returning `[]`) and may contain any Unicode characters, including two-, three-, and four-byte UTF-8 characters.
- `merges` may be empty. In that case, `encode` returns the raw UTF-8 byte values.
- `left` and `right` may be any id in `range(256 + len(merges))`. A table produced by BPE training only refers to earlier merges (`left, right < 256 + i`), but that is **not guaranteed here**. A merge may refer to a token created by a *later* merge, and the steps above apply unchanged. No token contains itself directly or indirectly, so every token expands to a finite byte string.
- For `decode`, every id is in `range(256 + len(merges))`. The list `ids` may be empty and need not be the output of `encode`.
- Both functions must return new objects without modifying their arguments. `encode` must return a list of Python `int` values.

Write the algorithm yourself without using a tokenizer library.

## Example

Let `a = 97`, `b = 98`, `c = 99`, and `d = 100`.

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

In the first step, each run `a a a` becomes `[aa] a` rather than `a [aa]` because the pass scans left to right. The pair `(a, b)` is present from the start, but it waits for rank 0 to run first.

Rank decides, not position:

```python
encode("abc", [(98, 99), (97, 98)])            # [97, 256]   (b, c) has rank 0
encode("abc", [(97, 98), (98, 99)])            # [256, 99]   (a, b) has rank 0
```

A merge may depend on a later merge. Below, token 256 is `(257, c)` and token 257 is `(a, b)`. Only rank 1 applies at first, and applying it makes rank 0 applicable:

```python
encode("abc", [(257, 99), (97, 98)])           # [256]
```

Encoding starts from UTF-8 bytes, not characters. `"é"` encodes as the two bytes `195, 169`:

```python
encode("é", [])                                # [195, 169]
encode("héé", [(195, 169), (104, 256)])        # [257, 256]
decode([104, 195], [])                         # "h�"   a truncated character
```

## Constraints

- `len(text) <= 2000` characters and `len(merges) <= 1000`.
- Token id lists and decoded strings are compared for exact equality.
- There is no speed requirement beyond the time limit. The direct algorithm, one scan of the sequence per applied merge, is fast enough.
- `decode(encode(s, merges), merges) == s` must hold for every string `s` and every valid merge table. Tests verify this property on random Unicode strings.

## Why this matters

The tokenizer is the first stage of every inference request, and a language model only understands the exact token ids it saw during training. An encoder that applies merges in a slightly different order still produces ids that decode back to the original text, so nothing crashes. Instead, the model sees an unfamiliar segmentation and quietly degrades in quality. This bug is a common cause of quality gaps when porting a reference tokenizer to a new inference runtime.

On the output side, a streaming server decodes tokens as they arrive. Multi-byte characters routinely split across token boundaries, so a decoder that converts each token independently emits replacement characters in the middle of words.
