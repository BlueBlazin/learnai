The alignment and the stopping rule are exactly those of greedy verification: row `i`
belongs to draft token `i`, the first failure ends the accepted prefix, and one extra
row supplies the next token. Only the test changed, from "is the argmax" to "has a
probability above a per-row threshold". What does each row need, besides the
probability of its draft token, to compute that threshold?

---

Compute the entropy of all `K + 1` rows at once, in nats: `-(p * log(p)).sum(-1)`.
That expression is NaN for a row with a zero in it, because `0 * log(0)` is
`0 * -inf`. Feed the logarithm a harmless `1` wherever `p` is `0` (a `where`), so the
term becomes `0 * 0`. The threshold of a row is the **smaller** of `epsilon` and
`delta * exp(-entropy)`.

Read the probability of each draft token with one indexing operation: rows
`0..K-1`, columns `draft_tokens`.

---

With `prob` of shape `(K,)` and `threshold` of shape `(K + 1,)`:

```python
accepted = prob > threshold[:K]  # strict
n_accepted = cumprod(accepted).sum()  # 1 up to the first False, then 0
next_token = argmax(p[n_accepted])  # the target's greedy token at that row
```

Convert both to Python ints before returning.
