Separate the two rules and decide what shape each one has before combining them. The
causal rule compares a query index with a key index and is the same for every row of
the batch. The padding rule depends on the row and on one position only. Is that
position the query or the key? Ask what must not be read: a real token must never read
from a padding position.

---

Build three boolean tensors from one `arange(T)` and let broadcasting combine them:

- `real`, shape `(B, T)`: compare the positions with `lengths[:, None]`. For right
  padding the real positions are the first `lengths[b]`; for left padding, the last.
- `causal`, shape `(T, T)`: entry `[i, j]` is true when `j <= i`.
- the diagonal, shape `(T, T)`: entry `[i, j]` is true when `j == i`.

`real` describes keys, and keys are the **last** axis of the mask, so it enters as
`real[:, None, :]`, shape `(B, 1, T)`. Then count the allowed entries in each row of
the result for a left-padded example: none of them may be zero.

---

```
allowed = (causal[None, :, :] & real[:, None, :]) | diagonal      # (B, T, T), boolean
mask    = where(allowed, 0.0, -inf)                               # float32
result  = mask[:, None, :, :]                                     # (B, 1, T, T)
```

The diagonal is OR-ed in after the AND, so that it survives for padding positions.
Fill with a true `-inf`, and convert from boolean to float last: multiplying a boolean
by `-inf` gives `0 * -inf = NaN` in the allowed entries.
