Implement the encoder of a unigram tokenizer, the model behind SentencePiece. Each vocabulary piece has a log-probability score, and the encoder finds the segmentation with the highest total score (the Viterbi search) while handling unknown characters and exact score ties.

![This problem is the Encode step, choosing the split of the text with the highest total score.](assets/map.svg)

## Specification

```python
def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]: ...
```

- `text`: the string to split. Positions are **characters** (Unicode code points), not bytes. Do not normalize the string, and treat spaces like any other character.
- `pieces`: a non-empty dict mapping each vocabulary piece (a non-empty string) to its **score** (log-probability).
- Returns the list of pieces in order. Their concatenation equals `text`.

**Segmentations.** A segmentation splits `text` into consecutive non-empty substrings $x_1, \dots, x_m$. Each substring $x_i$ must be either:

- a **vocabulary piece**, which is a key in `pieces` with its corresponding score, or
- an **unknown character**, which is a single character not present as a key in `pieces`. Its score is

$$
s_{\text{unk}} = \min_{p \,\in\, \texttt{pieces}} \operatorname{score}(p) - 10 .
$$

An unknown character is allowed at any position whose single character is not in `pieces`, even if a longer vocabulary piece starts at that position. A character that is a key in `pieces` is never treated as unknown. Because every character can be taken individually, every string has at least one valid segmentation.

**Optimal segmentation.** Choose the best segmentation using these rules in order:

1. Highest **total score**, $\sum_{i=1}^{m} \operatorname{score}(x_i)$.
2. Among ties, the **fewest pieces** ($m$).
3. Among ties, the segmentation whose first piece is **longest**. If the first pieces have the same length, prefer the longest second piece, and so on (lexicographically largest list of piece lengths).

Any two distinct segmentations of the same string differ in piece length at some position, so these three rules define a unique answer.

An unknown character appears in the output as itself (a one-character string). An empty string returns `[]`. Do not modify `pieces`.

Write the search yourself using only the Python standard library. Do not use a tokenizer library.

## Example

```python
pieces = {
    "h": -3.0,
    "u": -3.0,
    "g": -3.0,
    "s": -2.0,
    "hu": -4.0,
    "ug": -3.5,
    "gs": -4.5,
    "hug": -9.0,
}

unigram_encode("hugs", pieces)  # ["hu", "gs"]
unigram_encode("hug", pieces)  # ["h", "ug"]
unigram_encode("hugz", pieces)  # ["h", "ug", "z"]
```

![Every piece that matches `hugs` is an arrow between two character positions, labeled with its score. A segmentation is a path from 0 to 4.](assets/lattice.svg)

Here are all six segmentations of `hugs`:

| segmentation | total | pieces |
|---|---|---|
| `hu gs` | $-4 - 4.5 = -8.5$ | 2 |
| `h ug s` | $-3 - 3.5 - 2 = -8.5$ | 3 |
| `hu g s` | $-9$ | 3 |
| `h u gs` | $-10.5$ | 3 |
| `hug s` | $-9 - 2 = -11$ | 2 |
| `h u g s` | $-11$ | 4 |

Both `hu gs` and `h ug s` reach $-8.5$, and `hu gs` wins because it uses fewer pieces. Although `hug` is the longest piece at position 0, its score is low, so the optimal segmentation avoids it.

In `hugz`, the character `z` is not in `pieces`. The lowest vocabulary score is $-9$, so an unknown character scores $-19$. The best segmentation splits `hug` into `["h", "ug"]` and appends `z`, giving a total of $-6.5 - 19 = -25.5$.

Rule 3 breaks ties when two segmentations have the same total score and the same number of pieces:

```python
unigram_encode("abc", {"a": -1.0, "c": -1.0, "ab": -2.0, "bc": -2.0})  # ["ab", "c"]
```

Both `ab c` and `a bc` total $-3$ with two pieces. `ab c` wins because its first piece `ab` is longer than `a`.

An unknown character can be optimal even when a vocabulary piece starts at the same position:

```python
unigram_encode("abcd", {"ab": -1.0, "bcd": -1.0})  # ["a", "bcd"]
```

No single character is in `pieces`, so each character is available as an unknown at $-1 - 10 = -11$. Starting with `ab` leaves `c` and `d` as unknowns for a total of $-23$. Starting with the unknown `a` followed by `bcd` scores $-11 - 1 = -12$, which is higher.

## Constraints

- `text` has at most 2,000 characters. `pieces` has 1 to 2,000 entries, and each piece is 1 to 32 characters long.
- Every score is finite, at most 0, at least $-1000$, and a multiple of $1/8$. Sums of these numbers are exact in floating point, so equal totals match regardless of addition order.
- Results are compared exactly, piece by piece. On texts of up to 13 characters, results are also checked against an exhaustive search over all segmentations.
- There is no speed requirement beyond the time limit. However, a string of length $n$ has $2^{n-1}$ segmentations, so exhaustive enumeration will not finish at $n = 2000$.

## Why this matters

Unigram is one of the two major subword tokenizer families, used by SentencePiece models such as T5, ALBERT, XLNet, and many multilingual and speech models. Any inference engine serving these models must reproduce the exact Viterbi search.

Small implementation shortcuts fail silently. Taking the longest match at each position agrees with Viterbi on most common words, then diverges on rare ones and feeds the model valid token ids in an unexpected order. Treating unknown characters as free or fatal alters the segmentation of surrounding text. Indexing bytes instead of code points works on ASCII and breaks on the first unknown accented letter or emoji. None of these bugs raise an error, yet all of them degrade model output.
