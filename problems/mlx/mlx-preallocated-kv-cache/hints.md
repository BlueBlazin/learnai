An update falls into one of two cases: the new tokens fit in the existing buffers, or the buffers need to grow. For each case, work out which arrays must exist afterward and what their shapes should be. Write the capacity formula in code first and verify it against all four rows of the worked example, paying close attention to how many slots the old buffer contributes when growing (`offset`, not its full capacity).

---

Save `previous = self.offset` at the start of the update. To grow the cache, allocate `ceil(T / step) * step` slots of zeros matching the incoming dtypes (`mx.zeros(shape, keys.dtype)`, since the default is float32). If buffers already exist, concatenate only the used prefix (`[..., :previous, :]`) of each buffer with the new zeros along axis 2. Otherwise, use the new zeros directly as the buffers.

In both cases, write the incoming tokens with slice assignment (which preserves the buffer shape) and return only the filled prefix:

```python
self.keys[..., previous : self.offset, :] = keys
return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
```

You can compute integer ceiling division without floats using `-(-T // step)` or `(T + step - 1) // step`.

---

Here is the complete update logic:

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

Follow this block with the two slice assignments and the sliced return statement from the previous hint. Do not call `mx.eval` anywhere in the method so every operation remains part of the caller's lazy graph.
