In `x[i, j]` with two integer tensors, the result has the shape that `i` and `j`
broadcast to. For each function, write down the shape of the result you want, leaving
out any axis you do not index. What shapes must the two index tensors have so that
they broadcast to it?

---

- `pick_pairs`: the result is `(N,)`, and both index tensors already have that shape.
- `pick_block`: the result is `(R, C)`. `(R,)` and `(C,)` never broadcast to that: they
  broadcast to one axis when `R == C` or one of them is 1, and not at all otherwise. A column `(R, 1)` and a row `(1, C)` do.
- `pick_per_row`: the indexed part of the result is `(B, K)`, the shape of `cols`. The
  step index is `cols` itself. You also need a row index of a shape that broadcasts
  with `(B, K)` and whose entry `[b, k]` is `b`.

---

The key lines:

```
x[rows, cols]                       # (N,) with (N,)      -> (N,)

x[rows[:, None], cols[None, :]]     # (R, 1) with (1, C)  -> (R, C)

batch = arange(B)
x[batch[:, None], cols]             # (B, 1) with (B, K)  -> (B, K), and D is kept: (B, K, D)
```
