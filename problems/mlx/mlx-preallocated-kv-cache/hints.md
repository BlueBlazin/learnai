There are only two moments in this cache's life: an update that fits, and one that
does not. For each, which arrays must exist afterwards, and what are their shapes?
Write the capacity rule as code first and check it against the four rows of the
example; note in particular what the old buffer contributes when it grows (its first
`offset` slots, not all of its slots).

---

Growth: make zeros for the new room, `ceil(T / step) * step` slots, with the dtype
of the incoming arrays (`mx.zeros(shape, keys.dtype)`; the default is float32). If
buffers exist, concatenate the *used part* of each old buffer with the new room along
axis 2; if not, the new room is the buffer. Then, in both cases, write with slice
assignment, which keeps the buffer's shape, and return a slice:

```python
self.keys[..., previous : self.offset, :] = keys
return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
```

Integer ceiling without floats: `-(-T // step)`, or `(T + step - 1) // step`.

---

The whole method:

```python
previous, new = self.offset, keys.shape[2]
if self.keys is None or previous + new > self.keys.shape[2]:
    B, H, _, D_k = keys.shape
    room = -(-new // self.step) * self.step
    more_k = mx.zeros((B, H, room, D_k), keys.dtype)
    more_v = mx.zeros((B, H, room, values.shape[3]), values.dtype)
    if self.keys is None:
        self.keys, self.values = more_k, more_v
    else:
        self.keys = mx.concatenate([self.keys[..., :previous, :], more_k], axis=2)
        self.values = mx.concatenate([self.values[..., :previous, :], more_v], axis=2)
self.offset = previous + new
```

followed by the two slice assignments and the return above. Nothing here calls
`mx.eval`: every line adds nodes to the graph of the forward pass that called it.
