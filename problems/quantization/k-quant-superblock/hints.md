Write `q4k_decode` first because it defines the exact grid the format represents, and the encoder's final step is to pick codes for that grid. Notice that the encoder performs two levels of quantization. Level 1 summarizes each block of 32 weights with a float scale and minimum, and level 2 quantizes the 8 scales and 8 minimums into 6-bit integers plus two float16 units. Which level must finish before you compute the 4-bit weight codes?

---

For `q4k_decode`, cast all inputs to float32 before multiplying, because a block scale of 63 times a code of 15 overflows `uint8`. Reshape `codes` to `(S, 8, 32)` so the per-block step and offset of shape `(S, 8)` broadcast cleanly along the last axis.

For `q4k_encode`, reshape `w` to `(S, 8, 32)` and compute `lo = minimum(min(blocks, -1), 0)`, `hi = max(blocks, -1)`, `block_scale = (hi - lo) / 15`, and `block_min = -lo`. Take the maximum of each over the 8 blocks with `keepdim=True`. Compute the 6-bit fields as `round(63 * value / largest)`, using `where` to return `0` when `largest == 0`. Compute `d` and `dmin` as `largest / 63` cast to float16.

---

Compute the 4-bit codes last, using the exact step and offset that `q4k_decode` will reconstruct:

```python
step = d.astype(float32) * scales  # d is already float16 here
offset = dmin.astype(float32) * mins
safe = where(step > 0, step, 1)
codes = clip(round((blocks + offset[..., None]) / safe[..., None]), 0, 15)
codes = where(step[..., None] > 0, codes, 0)
```

Using the level-1 `block_scale` and `block_min`, or using float32 `d` and `dmin` before casting to float16, produces codes for a grid that the decoder will never reconstruct. Cast `codes`, `scales`, and `mins` to `uint8` at the end, and return `d` and `dmin` with shape `(S,)`.
