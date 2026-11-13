The merge list is not a to-do list to walk through once, and it is not a dictionary of
substrings to search for. At every step the question is the same: of the adjacent
pairs that are in the sequence *right now*, which one has the lowest rank? What data
structure answers "what is the rank of this pair?" in constant time?

---

Build a dictionary from pair to rank. Start with `list(text.encode("utf-8"))`. Loop:
look at every adjacent pair of the current sequence, pick the one with the smallest
rank (pairs that are not in the dictionary lose to all that are), and stop if not even
the best pair is in the dictionary. Otherwise rebuild the sequence with a `while` loop
over an index `i`: if positions `i` and `i + 1` form the chosen pair, append the merged
id and advance by **two**; otherwise append `ids[i]` and advance by one. Advancing by
two is what makes `a a a` come out as `[aa] a`.

---

The selection is one line:

```python
pair = min(pairwise(ids), key=lambda p: rank.get(p, float("inf")))
```

(`pairwise` is in `itertools`; it yields each adjacent pair) and the merged id is
`256 + rank[pair]`. For `decode`, expand each id into bytes: an id
below 256 is one byte, and any other id expands to the bytes of `left` followed by the
bytes of `right` from `merges[id - 256]` (recursively, or with a stack, since `left`
and `right` can be merged tokens themselves). Collect all bytes of all tokens into one
`bytearray` first and call `.decode("utf-8", errors="replace")` once at the end.
