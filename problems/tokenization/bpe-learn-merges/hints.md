Keep the corpus in its frequency-map form, with one entry per distinct word and its count. Think about how to represent each word so that counting pairs, applying a merge, and recounting are each a simple loop. Which of those steps must run again after every merge?

---

Store each word as a list of symbols, `[*word, "</w>"]`, paired with its frequency. In each step, build a `Counter` of adjacent pairs across all words using `itertools.pairwise` (which includes overlapping pairs) and add the word's frequency for each pair. If the counter is empty, stop. Otherwise, pick the winning pair, append it to the result, and rebuild every word's symbol list with that pair merged. Recount from the rebuilt lists on the next step because every merge destroys old pairs and creates new ones.

---

Selection with the tie-breaking rule is a single expression that prioritizes the highest count and then the smallest pair:

```python
best = min(counts, key=lambda pair: (-counts[pair], pair))
```

Apply the merge with an index loop rather than string replacement. At position `i`, if `symbols[i] == left` and `symbols[i + 1] == right`, append `left + right` and advance `i` by **two**. Otherwise, append `symbols[i]` and advance `i` by one. Advancing by two turns `a a a` into `aa a`, and comparing whole list elements keeps `(a, b)` from matching inside `xa b`.
