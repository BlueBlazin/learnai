Draw the request as a timeline: the send time, then one mark per token. Which metrics
measure from the send time, and which measure only between marks? How many gaps are
there between `n` marks?

---

`ttft` and `e2e` start at `t_start` and end at the first and the last token. `itl` is
the list of differences between neighbors, `token_times[i] - token_times[i - 1]` for
`i` from 1, so it has `n - 1` entries and never involves `t_start`. `tpot` is the mean
of those gaps; their sum telescopes to `token_times[-1] - token_times[0]`, so no loop
is needed. `tokens_per_s` is the only metric that counts all `n` tokens, and it divides
by the whole wait, `e2e`. Handle the empty list first, then the two divisions that can
have a zero denominator.

---

```python
first, last = token_times[0], token_times[-1]
e2e = last - t_start
tpot = (last - first) / (n - 1) if n > 1 else None
tokens_per_s = n / e2e if e2e > 0 else None
```

The time *and* the count in `tpot` both leave the first token out: `n - 1` gaps over
the time from the first token to the last. Raise `ValueError` when `n == 0`, and build
`itl` as a new list so that `token_times` stays untouched.
