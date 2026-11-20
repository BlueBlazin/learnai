The definition never mentions calls: the windows are fixed by the row numbers of the
stream. So ask what a call must know about the stream that its own arguments do not
tell it. Which rows of the current window have already arrived, and where in a window
does the first new row fall?

---

Put the waiting rows in front of the new ones (they are older) and forget the call
boundary:

```python
kv = concatenate((buf_kv, kv))
gate = concatenate((buf_gate, gate))
```

Row 0 of this tensor is at offset 0 of a window, because the buffer always starts at
a window boundary. So the first `m = len(kv) // rate` windows are simply consecutive
blocks of `rate` rows, and whatever is left after `m * rate` rows is the new buffer.
A reshape to `(m, rate, d)` makes the window an axis, and `position_bias`, of shape
`(rate, d)`, then broadcasts onto every window with the right offsets.

---

```python
win_kv = kv[: m * rate].reshape(m, rate, d)
win_gate = gate[: m * rate].reshape(m, rate, d) + position_bias
pooled = (softmax(win_gate, axis=1) * win_kv).sum(axis=1)  # (m, d)
entries = gain * pooled * rsqrt(mean(pooled**2, axis=-1, keepdims=True) + 1e-6)
new_state = (kv[m * rate :], gate[m * rate :], count + m)
```

The softmax axis is the window axis (1), not the channel axis. With `m = 0`
everything above still works on empty tensors and returns shape `(0, d)`.

For `visible_entries`: the last row of window `w` is at position
`(w + 1) * rate - 1`. Count the `w >= 0` for which that is at most `t`.
