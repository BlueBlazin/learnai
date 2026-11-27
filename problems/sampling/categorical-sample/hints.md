Picture the interval $[0, 1)$ cut into consecutive pieces, one per token, where
piece $k$ has length $p_k$. A uniform number lands in piece $k$ with probability
$p_k$, and the right endpoints of the pieces are the cumulative sums. Consider two
edge cases: how wide is the piece for a token with probability 0, and where does the
last piece end if the probabilities sum to 0.999?

---

Compute the cumulative sum along the last axis. To make the pieces cover the full
range that `u` can reach, either divide by the row total or, equivalently and more
cheaply, multiply `u` by the row total. Read the row total from the last column of
the cumulative sum rather than a separate `sum`, because the two can differ in the
last bit and you need the threshold to stay strictly below the last cumulative sum.
Give `u` (shape `(B,)`) a trailing axis so each row compares against its own
threshold.

---

Because `cdf` is non-decreasing, the first index whose cumulative sum exceeds the
threshold equals the count of entries that do not exceed it:

```
cdf = cumsum(probs, axis=-1)
threshold = u[:, None] * cdf[:, -1:]
token = sum(cdf <= threshold, axis=-1)
```

Check this against the worked example. A zero-probability token has the same
cumulative sum as its predecessor, so `<=` counts both or neither and the result can
never land on it. Because `threshold < cdf[:, -1]`, the last column is never
counted, so the returned index is always at most `V - 1`.
