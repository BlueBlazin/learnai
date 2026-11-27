The best segmentation of a string starts with some first piece and continues with a segmentation of the remaining suffix. Can that continuation be anything other than the best segmentation of the suffix? If not, how many distinct suffixes does a string of $n$ characters have?

---

A string of length $n$ has only $n + 1$ suffixes, `text[i:]` for `i` from `0` to `n`. Work backwards from `n - 1` down to `0`, computing the best segmentation of each suffix as a tuple `(total_score, piece_count, first_piece_length)`.

At position `i`, try every length `L` up to the longest piece in `pieces`. A length `L` is valid if `text[i:i+L]` is in `pieces`, or if `L == 1` and `text[i]` is not in `pieces` (the unknown case, scored at the minimum piece score minus 10). Combine that piece with the stored result for `i + L`. Once the table is filled, walk forward from position `0` using the stored first-piece lengths.

Working from the end makes the third tie-breaking rule local. That rule compares the *first* piece, which is exactly the piece you are choosing at position `i`.

---

A candidate at position `i` with length `L` has:

```python
total = score + best[i + L].total
count = 1 + best[i + L].count
```

Iterate `L` from the longest possible length down to `1`. Replace the current best choice only when `total` is higher, or when `total` is equal and `count` is lower. A candidate that ties on both loses to the earlier choice, which has the longer first piece. Because `best[i + L]` is already optimal under all three rules, the choice at `i` is optimal too. Include the unknown character at `L == 1` whenever `text[i]` is not a key in `pieces`, regardless of whether longer pieces start at `i`.
