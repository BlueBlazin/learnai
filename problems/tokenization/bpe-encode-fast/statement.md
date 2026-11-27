Encode text with byte-level Byte Pair Encoding (BPE) in $O(n \log n)$ time, where $n$ is the byte length of the input text. Your function must return the exact same token ids as `bpe-encode`. The naive definition rescans the entire sequence for every merge it applies, which takes quadratic time on long inputs with deep merge tables. A faster algorithm must avoid those full scans while reproducing the exact merge order, including ties.

![This problem is the Encode step, producing the ids of `bpe-encode` in $O(n \log n)$ time.](assets/map.svg)

## Specification

```python
def encode(text: str, merges: list[tuple[int, int]]) -> list[int]: ...
```

**Token ids.** Ids `0` through `255` represent the 256 possible byte values. An entry `merges[i] = (left, right)` specifies that token `left` immediately followed by token `right` merges into token `256 + i`. The index `i` is the merge's **rank**, where a lower rank means higher priority.

**`encode(text, merges)`** returns the list of token ids produced by the following reference procedure. Your implementation must return the exact same list, though it should compute the result more efficiently.

1. Convert `text` to its UTF-8 bytes, one token id per byte.
2. Find all adjacent pairs in the current sequence that appear in `merges`. If none exist, stop and return the current sequence.
3. Choose the matching pair with the lowest rank. Neither position in the text nor frequency affects this choice.
4. Replace occurrences of that pair with `256 + rank` in a single left-to-right pass. When the pair appears at positions `j` and `j + 1`, emit the merged id and continue at `j + 2`. Replacements never overlap, and in a run of repeated tokens the leftmost matches win. For example, merging `(a, a)` turns `a a a` into `aa a` and `a a a a` into `aa aa`.
5. Repeat from step 2 on the updated sequence.

A faster algorithm must preserve three properties of this procedure:

- **Finish the full pass before choosing again.** Step 4 completes its pass over the whole sequence before step 2 inspects the sequence again. A new pair created during the pass can only be chosen in a later step, even if its rank is lower than the rank currently being applied.
- **Merge only pairs that still exist.** If a pair appeared earlier in the sequence but one of its tokens was consumed by another merge, it cannot be merged.
- **Process each pass left to right.** Within a single rank's pass, overlapping occurrences are resolved from the left.

Additional details:

- `text` may be empty (return `[]`) and may contain any Unicode characters.
- `merges` may be empty. All pairs in `merges` are distinct.
- `left` and `right` may be any id in `range(256 + len(merges))`. A table from standard BPE training only references earlier merges (`left, right < 256 + i`), but that is **not guaranteed here**. A merge may reference a token created by a *later* merge. No token contains itself, either directly or indirectly.
- Return a new list of Python `int` values without modifying `merges`.

Implement the algorithm yourself using only the Python standard library. Do not use a tokenizer library.

## Example

Let `a = 97`, `b = 98`, `c = 99`, and `d = 100`.

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

Each pass must finish before the next rank is chosen. With `merges = [(257, 97), (97, 98)]`, token `257` is `ab` and rank 0 is `([ab], a)`:

```python
encode("abab", [(257, 97), (97, 98)])  # [257, 257]
```

At the start, only `(a, b)` (rank 1) is present. Its pass turns `a b a b` into `[ab] [ab]`. Immediately after the first replacement, the intermediate sequence `[ab] a b` contains the rank-0 pair `([ab], a)`, but the rank-1 pass has not finished yet. Once the pass finishes, that pair no longer exists. Merging it early would produce `[aba] b`, or `[256, 98]`.

A pair can also disappear before its turn:

```python
encode("abc", [(98, 99), (97, 98)])  # [97, 256]
```

Both `(a, b)` and `(b, c)` are present initially, but `(b, c)` has the lower rank and consumes `b` first.

## Constraints

- `text` has at most 50,000 bytes of UTF-8, and `len(merges) <= 50,000`.
- Token ids are compared exactly.
- **Speed.** One benchmark tests your function against the reference on a 2,000-byte text with about 1,900 merges, constructed so that almost every merge applies once. Your solution must run within **5x** of the reference. On this input, the literal scan procedure rescans up to 2,000 tokens roughly 1,900 times and measures about 200x slower than the reference. A variant that speeds up merge selection but still rebuilds the sequence measures about 140x slower. An $O(n \log n)$ implementation using standard Python data structures typically measures between 1x and 2x.
- All other test cases have no speed requirement beyond the 10-second time limit, with texts up to 50,000 bytes.

## Why this matters

Scanning the sequence once per merge works fine when inputs are short 10-byte words produced by whitespace pre-tokenization on English prose. It breaks down on real serving traffic such as base64 payloads, minified JavaScript bundles, long stretches of CJK text without spaces, or DNA sequences. Those inputs form single pieces tens of kilobytes long, where a quadratic encoder burns seconds of CPU time on the request path before the model ever runs.

Because an attacker can craft such prompts, quadratic tokenization is also a denial-of-service vector. Production tokenizers avoid it by encoding long pieces with a priority queue over a linked sequence while matching the simple definition id for id, since the model was trained on that exact tokenization.
