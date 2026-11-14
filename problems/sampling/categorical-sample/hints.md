Picture the interval $[0, 1)$ cut into consecutive pieces, one per token, where piece
$k$ has length $p_k$. A uniform number lands in piece $k$ with probability $p_k$. The
right ends of the pieces are the cumulative sums. Two questions: what is the length of
the piece of a token with probability 0, and where does the last piece end if the
probabilities sum to 0.999?

---

Compute the cumulative sum along the last axis. To make the pieces cover exactly the
range the uniform number can reach, either divide by the row total or, equivalently
and cheaper, multiply `u` by it. Take the total from the last column of the
cumulative sum, not from a separate `sum`: the two can differ in the last bit, and
the argument "the threshold is below the last cumulative sum" needs them to be the
same number. `u` has shape `(B,)`; give it a trailing axis so that each row is
compared with its own threshold.

---

The first index whose cumulative sum exceeds the threshold is the number of entries
that do not exceed it:

```
cdf = cumsum(probs, axis=-1)
threshold = u[:, None] * cdf[:, -1:]
token = sum(cdf <= threshold, axis=-1)
```

Check it against the worked example. A zero-probability token has the same cumulative
sum as its predecessor, so `<=` counts both or neither, and the count can never stop
on it. And since `threshold < cdf[:, -1]`, the last column is never counted, so the
result is at most `V - 1`.
