For each function, write down the shape after every operation twice: once for a general input and once when `B = 1`. If the two results have different numbers of axes, one of your operations is dropping an axis based on its runtime size rather than its position.

---

- `normalize_over`: summing over `dim` removes that axis, so the result has one fewer dimension than `x`. To divide cleanly for any `dim`, tell the reduction to keep the summed axis with size 1.
- `last_step`: you want to drop the `T` axis and keep `B` and `D`. Which kind of index removes only the axis it indexes?
- `column`: you want to keep the `V` axis with size 1. Which kind of index preserves its axis?

---

The key lines:

```
x / sum(x, over dim, keeping the axis)      # keepdim=True in PyTorch, keepdims=True in JAX and MLX

x[:, -1]                                    # (B, T, D) -> (B, D): an integer removes its axis

x[:, j : j + 1]                             # (B, V) -> (B, 1): a slice keeps its axis
```

`x[:, -1:].squeeze(1)` and `x[:, j][:, None]` are equally valid. Calling `squeeze()` with no argument is not.
