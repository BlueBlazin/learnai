Represent each word as a tuple of symbols. A `Counter` keyed by pairs does the counting.
---
`max(counts, key=lambda pair: (counts[pair], ...))` needs a tie-break that prefers the smaller pair while maximizing the count. Negating a string is not possible; use `min` with the key `(-count, pair)` instead.
