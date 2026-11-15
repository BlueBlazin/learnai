Keep the corpus as it is given: one entry per distinct word, with its frequency. What
do you need to store for each word so that "count the pairs", "apply a merge", and
"count again" are each a simple loop? And which of those three has to be repeated
after every single merge?

---

Represent each word as a list of symbols, `[*word, "</w>"]`, next to its frequency.
One step is: build a `Counter` of adjacent pairs (`itertools.pairwise` yields them,
overlapping ones included; add the word's frequency for each), stop if it is empty,
choose a pair, append it to the result, and rebuild every word's list with the pair
merged. The counts are taken again from the rebuilt lists in the next step: a merge
destroys some pairs and creates new ones.

---

Selection with the tie rule is one expression: the highest count first, then the
smallest pair.

```python
best = min(counts, key=lambda pair: (-counts[pair], pair))
```

Apply the merge with an index loop, not with string replacement: at position `i`, if
`symbols[i] == left` and `symbols[i + 1] == right`, append `left + right` and advance
by **two**; otherwise append `symbols[i]` and advance by one. Advancing by two is what
turns `a a a` into `aa a`, and comparing whole list elements is what keeps `(a, b)`
from matching inside `xa b`.
