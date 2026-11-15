Split the function into three per-row questions, each answered by a boolean: which
tokens are top choices, does XTC act on this row, and which single top choice
survives. The output is then one `where`. Which of the three needs the softmax, and
which can be answered from the logits alone?

---

`above = softmax(logits) >= threshold` marks the top choices. XTC acts on a row when
`above` has at least two true entries **and** `u[b] < probability`; give both a
trailing axis of length 1 so they broadcast over the row. The survivor is the top
choice with the smallest logit: mask the other entries with `+inf` and take the
row minimum. No sort is needed, because the top choices are exactly the tokens above
a cutoff, and you only need the smallest of them.

---

```
above = softmax(logits) >= threshold
fires = (count(above) >= 2) & (u[:, None] < probability)
floor = min(where(above, logits, +inf), axis=-1, keepdims=True)
candidate = above & (logits == floor)
survivor = max(where(candidate, arange(V), -1), axis=-1, keepdims=True)
remove = fires & above & (arange(V) != survivor)
out = where(remove, -inf, logits)
```

`candidate` can mark several tokens when logits tie; taking the largest index among
them picks exactly one. Compare logits, not probabilities, to find the ties: equal
logits are equal bit for bit, while their computed probabilities need not be.
