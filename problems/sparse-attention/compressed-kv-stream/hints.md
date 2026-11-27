Windows are fixed by row numbers in the overall stream, not by call boundaries. Ask what a call needs from earlier calls that its own `kv` and `gate` arguments do not provide. Specifically, which rows of the current unfinished window have already arrived, and where inside that window does the first new row belong?

---

Place the buffered rows in front of the new rows (since they are older) so the call boundary disappears:

```python
kv = concatenate((buf_kv, kv))
gate = concatenate((buf_gate, gate))
```

Because the buffer always begins on a window boundary, row 0 of this combined tensor is at window offset 0. The first `m = len(kv) // rate` windows are simply consecutive blocks of `rate` rows, and any rows after `m * rate` become the new buffer. Reshaping the usable prefix to `(m, rate, d)` creates an explicit window axis so `position_bias` of shape `(rate, d)` broadcasts onto every window with the right offsets.

---

```python
win_kv = kv[: m * rate].reshape(m, rate, d)
win_gate = gate[: m * rate].reshape(m, rate, d) + position_bias
pooled = (softmax(win_gate, axis=1) * win_kv).sum(axis=1)  # (m, d)
entries = gain * pooled * rsqrt(mean(pooled**2, axis=-1, keepdims=True) + 1e-6)
new_state = (kv[m * rate :], gate[m * rate :], count + m)
```

Make sure the softmax runs over the window axis (`axis=1`), not the channel axis. When `m == 0`, these operations still work on empty tensors and return shape `(0, d)`.

For `visible_entries`, the last row of window `w` is at position `(w + 1) * rate - 1`. Count how many integers `w >= 0` satisfy `(w + 1) * rate - 1 <= t` using exact integer arithmetic.
