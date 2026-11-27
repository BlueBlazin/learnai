The merge list is not a checklist to walk through once, and it is not a dictionary of substrings to search for. At every step, ask which adjacent pair in the current sequence has the lowest rank. What data structure lets you look up the rank of a pair in constant time?

---

Build a dictionary mapping each `(left, right)` pair to its rank. Start with `ids = list(text.encode("utf-8"))`. In a loop, inspect every adjacent pair in `ids` and pick the one with the smallest rank, treating pairs outside the dictionary as worse than any valid merge. Stop when no adjacent pair is in the dictionary. Otherwise, rebuild `ids` with a `while` loop over an index `i`. When `ids[i]` and `ids[i + 1]` match the chosen pair, append the merged id and advance `i` by **two**. Otherwise, append `ids[i]` and advance `i` by one. Advancing by two ensures `a a a` becomes `[aa] a`.

---

The pair selection is one line:

```python
pair = min(pairwise(ids), key=lambda p: rank.get(p, float("inf")))
```

Here `pairwise` from `itertools` yields each adjacent pair, and the merged id is `256 + rank[pair]`. For `decode`, expand each id into bytes. An id below 256 is a single byte, while any higher id expands to the bytes of `left` followed by the bytes of `right` from `merges[id - 256]`. Use recursion or an explicit stack since `left` and `right` can be merged tokens themselves. Collect all bytes into one `bytearray`, then call `.decode("utf-8", errors="replace")` once at the end.
