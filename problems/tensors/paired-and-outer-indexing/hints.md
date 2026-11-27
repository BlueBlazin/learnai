In `x[i, j]` with two integer tensors, the output takes the shape that `i` and `j` broadcast to. For each function, write down the desired output shape while omitting any unindexed axes. What shapes must the two index tensors have so that they broadcast to it?

---

- `pick_pairs`: The output shape is `(N,)`, and both index tensors already have that shape.
- `pick_block`: The output shape is `(R, C)`. Two 1D tensors of shapes `(R,)` and `(C,)` never broadcast to `(R, C)`. They either broadcast to a single axis when `R == C` or one of them is 1, or fail to broadcast at all. A column `(R, 1)` and a row `(1, C)` broadcast to `(R, C)`.
- `pick_per_row`: The indexed part of the output has shape `(B, K)`, which matches `cols`. You also need a row index tensor whose shape broadcasts with `(B, K)` and whose entry at `[b, k]` equals `b`.

---

The key lines:

```
x[rows, cols]                       # (N,) with (N,)      -> (N,)

x[rows[:, None], cols[None, :]]     # (R, 1) with (1, C)  -> (R, C)

batch = arange(B)
x[batch[:, None], cols]             # (B, 1) with (B, K)  -> (B, K), and D is kept: (B, K, D)
```
