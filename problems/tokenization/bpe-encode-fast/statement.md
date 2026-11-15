Encode text with byte-level BPE in $O(n \log n)$ time. The result is the one that
`bpe-encode` defines, id for id; what changes is the cost. That definition rescans the
whole sequence for every merge it applies, which is quadratic on a long input with a
deep merge table, and a faster algorithm has to reproduce its order of merges exactly,
including the cases where the order is decided by a tie.

## Specification

```python
def encode(text: str, merges: list[tuple[int, int]]) -> list[int]: ...
```

**Token ids.** Ids `0` to `255` are the 256 possible byte values. `merges[i] = (left,
right)` says that the token `left` immediately followed by the token `right` merges
into the token with id `256 + i`. The position `i` is the merge's **rank**; a lower
rank means a higher priority.

**`encode(text, merges)`** returns the list of token ids that the following procedure
produces. Your function must return the same list; it does not have to work this way.

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

Three consequences that a faster algorithm must preserve:

- **All occurrences, then the next choice.** Step 4 finishes its pass over the whole
  sequence before step 2 looks at the sequence again. A pair that the pass creates
  takes part in the next choice, not in the current pass, even if its rank is lower
  than the rank being applied.
- **Only pairs that are present.** A pair that was in the sequence earlier and has
  since lost one of its two tokens to another merge is not merged.
- **Left to right.** Within one pass the occurrences are taken from the left.

Details:

- `text` may be empty (the result is `[]`) and may contain any Unicode characters.
- `merges` may be empty. Its pairs are distinct.
- `left` and `right` may be any id in `range(256 + len(merges))`. A table produced by
  BPE training only refers to earlier merges (`left, right < 256 + i`), but that is
  **not guaranteed here**: a merge may refer to a token that a *later* merge creates.
  No token contains itself, directly or through other merges.
- The function returns a new list of Python ints and does not modify `merges`.

Write the algorithm yourself, with the standard library only: do not use a tokenizer
library.

## Example

Let `a = 97`, `b = 98`, `c = 99`, `d = 100`.

```python
merges = [(97, 97), (97, 98), (256, 257)]  # 256 = "aa", 257 = "ab", 258 = "aaab"
encode("aaabdaaabac", merges)  # [258, 100, 258, 97, 99]
```

| sequence | pairs present that are merges | applied |
|---|---|---|
| `a a a b d a a a b a c` | `(a, a)` rank 0, `(a, b)` rank 1 | rank 0, twice |
| `[aa] a b d [aa] a b a c` | `(a, b)` rank 1 | rank 1, three times |
| `[aa] [ab] d [aa] [ab] [ab] c` | `([aa], [ab])` rank 2 | rank 2, twice |
| `[aaab] d [aaab] [ab] c` | none | stop |

Wait for the pass to finish. With `merges = [(257, 97), (97, 98)]`, token 257 is `ab`
and rank 0 is the pair `([ab], a)`:

```python
encode("abab", [(257, 97), (97, 98)])  # [257, 257]
```

The only merge present at the start is `(a, b)`, rank 1. Its pass turns `a b a b` into
`[ab] [ab]`. After the first replacement the sequence read `[ab] a b`, which contains
the rank 0 pair `([ab], a)`, but the pass was not finished, and when it is, that pair
no longer exists. Merging it early would give `[aba] b`, that is `[256, 98]`.

A pair can disappear before its turn:

```python
encode("abc", [(98, 99), (97, 98)])  # [97, 256]
```

`(a, b)` is present at the start, but `(b, c)` goes first and takes the `b`.

## Constraints

- `text` has at most 50,000 bytes of UTF-8, and `len(merges) <= 50,000`.
- Ids are compared exactly.
- **Speed.** One case times your function against the reference on a text of 2,000
  bytes with a table of about 1,900 merges, built so that almost every merge applies
  exactly once. Your function must be within **5x** of the reference. On that input
  the procedure above, implemented literally, scans a sequence of up to 2,000 tokens
  about 1,900 times; it was measured at 200 times slower than the reference, and a
  variant that only finds the next merge faster but still rebuilds the sequence at
  140 times slower. An $O(n \log n)$ solution with ordinary Python data structures
  measures between 1x and 2x.
- The other cases have no speed requirement beyond the time limit of 10 seconds, with
  texts of up to 50,000 bytes.

## Why this matters

A tokenizer that runs one scan per merge is fine while every input is a word of ten
bytes, which is what pre-tokenization on whitespace produces for English prose. It
stops being fine on the inputs a serving stack actually receives: a base64 blob, a
minified JavaScript bundle, a long line of CJK text without spaces, a DNA string.
Those are single "words" of tens of kilobytes, and a quadratic encoder turns one such
request into seconds of CPU time on the request path, before the model has seen a
token. It is also a denial-of-service vector: the cost is chosen by whoever writes the
prompt. Production tokenizers therefore encode each word with a priority queue over a
linked sequence, and they must still agree with the simple definition on every id,
because the model was trained on that definition's output.
