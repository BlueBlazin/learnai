Implement the encoder of a unigram tokenizer, the model behind SentencePiece: every
vocabulary piece has a log-probability, and a text is split into the sequence of
pieces with the highest total. There is no merge order to replay; the answer is the
optimum over all ways of cutting the text, found by dynamic programming. The difficulty
is in the corners that make the optimum unique: characters the vocabulary cannot spell,
and segmentations that score the same.

## Specification

```python
def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]: ...
```

- `text`: the string to split. Positions are **characters** (Unicode code points), not
  bytes. The text is taken as it is: no normalization, and a space is a character like
  any other.
- `pieces`: the vocabulary, a non-empty dict from piece (a non-empty string) to its
  **score**, the log-probability of the piece.
- Returns the list of pieces, in order. Their concatenation is `text`.

**Segmentations.** A segmentation cuts `text` into consecutive non-empty substrings
$x_1, \dots, x_m$. Each substring must be one of:

- a **vocabulary piece**: a key of `pieces`, with its score;
- an **unknown character**: a single character that is not a key of `pieces`, with
  the score

$$
s_{\text{unk}} = \min_{p \,\in\, \texttt{pieces}} \operatorname{score}(p) \;-\; 10 .
$$

  An unknown character is available wherever the character itself is not in the
  vocabulary, whether or not a longer piece starts at that position. A character that
  is in the vocabulary is never unknown.

Every text has at least one segmentation: one character at a time.

**The result** is the best segmentation in this order:

1. the highest **total score** $\sum_i \operatorname{score}(x_i)$;
2. among those, the **fewest pieces**;
3. among those, the one whose first piece is **longest**; if the first pieces are the
   same, the one whose second piece is longest; and so on. (The list of piece lengths
   is the largest in lexicographic order.)

Two different segmentations differ in their lengths at some position, so the result
is unique.

An unknown character appears in the result as itself, a one-character string. The
empty text gives `[]`. The function does not modify `pieces`.

Write the search yourself, with the standard library only: do not use a tokenizer
library.

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

The six segmentations of `hugs`:

| segmentation | total | pieces |
|---|---|---|
| `hu gs` | $-4 - 4.5 = -8.5$ | 2 |
| `h ug s` | $-3 - 3.5 - 2 = -8.5$ | 3 |
| `hu g s` | $-9$ | 3 |
| `h u gs` | $-10.5$ | 3 |
| `hug s` | $-9 - 2 = -11$ | 2 |
| `h u g s` | $-11$ | 4 |

Two reach $-8.5$, and `hu gs` has fewer pieces. The longest piece at the start, `hug`,
is in the vocabulary but rare, and the best segmentation does not use it.

In `hugz`, `z` is not in the vocabulary. The lowest score is $-9$, so an unknown
character scores $-19$, and the result is the best split of `hug` followed by `z`,
with the total $-6.5 - 19 = -25.5$.

Rule 3 decides between segmentations with the same total and the same number of
pieces:

```python
unigram_encode("abc", {"a": -1.0, "c": -1.0, "ab": -2.0, "bc": -2.0})  # ["ab", "c"]
```

`ab c` and `a bc` both total $-3$ with two pieces; `ab` is the longer first piece.

An unknown character can be the right choice even where a piece matches:

```python
unigram_encode("abcd", {"ab": -1.0, "bcd": -1.0})  # ["a", "bcd"]
```

No single character is in the vocabulary, so each is available as an unknown at
$-1 - 10 = -11$. Starting with `ab` leaves `c` and `d` unknown, $-23$ in total;
starting with the unknown `a` costs $-11 - 1 = -12$.

## Constraints

- `text` has at most 2,000 characters. `pieces` has 1 to 2,000 entries, each piece
  1 to 32 characters long.
- Every score is finite, at most 0, at least $-1000$, and a multiple of $1/8$. Sums of
  such numbers are exact in floating point, so two segmentations tie exactly when
  their totals are equal, in whatever order you add.
- The result is compared exactly, piece by piece. On texts of up to 13 characters it
  is also compared with an exhaustive search over all segmentations.
- There is no speed requirement beyond the time limit, but the number of
  segmentations is $2^{n-1}$ for $n$ characters: enumerating them is not an option at
  $n = 2000$.

## Why this matters

Unigram is one of the two tokenizer families you will meet in checkpoints
(SentencePiece models for T5, ALBERT, XLNet, and many multilingual and speech
models), and an engine that ships such a model has to reproduce this search. The
places where ports go wrong are the ones this problem pins down. An encoder that takes
the longest match at each position agrees with Viterbi on most common words and
diverges on the rest, so the model receives valid ids in an order it rarely saw.
An encoder that treats unknown characters as free, or as fatal, changes the
segmentation of everything around them. And an encoder that indexes bytes works until
the first accented letter or emoji that is not in the vocabulary. None of these fail
loudly: the ids are all in range, and the output is merely worse.
