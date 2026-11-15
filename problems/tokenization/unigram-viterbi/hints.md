The best segmentation of a text starts with some first piece and continues with a
segmentation of what remains. Can the continuation be anything other than the best
segmentation of the remainder? If not, how many different "remainders" are there for a
text of $n$ characters?

---

There are only $n + 1$ remainders: the suffixes `text[i:]`. Compute, for every `i`
from the end of the text to the start, the best segmentation of the suffix that
starts at `i`, as a small record: its total score, its number of pieces, and the
length of its first piece. For position `i`, try every length `L` (up to the longest
piece in the vocabulary): if `text[i:i+L]` is a piece, or `L == 1` and the character
is not in the vocabulary (the unknown case, scored at the lowest score minus 10), the
candidate is that piece followed by the stored best of `i + L`. Then walk forward from
position 0, following the stored first-piece lengths.

Working from the end is what makes the third tie rule easy. It is stated in terms of
the *first* piece, and at position `i` the first piece is exactly the thing being
chosen.

---

A candidate at position `i` with length `L` has the key

```python
total = score + best[i + L].total
count = 1 + best[i + L].count
```

Try the lengths from the longest down to 1, and replace the current choice only if
the candidate is strictly better: `total` higher, or `total` equal and `count` lower.
A candidate that ties on both loses to the one already chosen, which has the longer
first piece. Because the stored record of `i + L` is itself the best under all three
rules, the choice at `i` is too. The unknown character is one more candidate at
`L == 1`, available only when `text[i]` is not a key of `pieces`; do not make it
depend on whether some longer piece matched.
