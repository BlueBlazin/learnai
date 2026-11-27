Draw the request on a timeline with `t_start` followed by one mark per token. Decide which metrics start at `t_start` and which measure only between token marks. Then count how many gaps exist between `n` marks.

---

`ttft` and `e2e` both start at `t_start` and end at the first and last tokens. `itl` holds the consecutive differences `token_times[i] - token_times[i - 1]` for `i` starting at 1, giving `n - 1` values without using `t_start`. `tpot` is the mean of those gaps, and their sum telescopes to `token_times[-1] - token_times[0]`. `tokens_per_s` counts all `n` tokens and divides by `e2e`. Check for an empty list first, then guard the two divisions whose denominators can be zero.

---

```python
first, last = token_times[0], token_times[-1]
e2e = last - t_start
tpot = (last - first) / (n - 1) if n > 1 else None
tokens_per_s = n / e2e if e2e > 0 else None
```

Both the numerator and denominator of `tpot` exclude the first token, using `n - 1` gaps over the span from `first` to `last`. Raise `ValueError` when `n == 0`, and build `itl` as a new list so `token_times` is not modified.
