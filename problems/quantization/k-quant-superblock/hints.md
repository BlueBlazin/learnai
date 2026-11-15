Write the decoder first: it is the specification of the format, and the encoder's
last step is "choose the codes that this decoder turns into the nearest values". For
the encoder, notice that there are two levels of quantization. Level 1 turns 32
weights into codes plus a float scale and minimum. Level 2 turns 8 float scales and
8 float minimums into 6-bit integers plus one float16 unit each. Which of the two
levels must be finished before the codes can be computed?

---

Decoder: widen everything to float32 before multiplying (a block scale of 63 times
a code of 15 does not fit in a byte). Reshape the codes to `(S, 8, 32)` so that the
per-block step and offset, of shape `(S, 8)`, broadcast with a trailing axis.

Encoder, in the same shape: `lo = minimum(min(blocks, -1), 0)`, `hi = max(blocks,
-1)`, block scale `(hi - lo) / 15`, block minimum `-lo`. Take the maximum of each
over the 8 blocks (keep the axis). The 6-bit fields are `round(63 * value /
largest)`, guarded by a `where` for a largest value of 0. The units are `largest /
63`, cast to float16.

---

The codes come last, from the values the decoder will see:

```python
step = d.astype(float32) * scales  # d is already float16 here
offset = dmin.astype(float32) * mins
safe = where(step > 0, step, 1)
codes = clip(round((blocks + offset[..., None]) / safe[..., None]), 0, 15)
codes = where(step[..., None] > 0, codes, 0)
```

Using the level-1 `block_scale` and `block_min` here, or the float32 units from
before the float16 cast, gives codes for a grid that no decoder will ever rebuild.
Cast the three integer fields to uint8 at the very end, and return `d` and `dmin`
with shape `(S,)`.
