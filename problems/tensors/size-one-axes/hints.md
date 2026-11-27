For each function, write the shape you get after every operation, and do it twice: once
for a general input, and once for an input where `B = 1`. If the two shapes differ in
their number of axes, one of your operations is removing an axis because of its size
and not because you named it.

---

- `normalize_over`: the sum over `dim` has one axis fewer than `x`. To divide, it has
  to line up with `x` again along every other axis, for any `dim`. Ask the reduction to
  keep the axis it sums over.
- `last_step`: you want to remove the `T` axis and no other. Which kind of index
  removes exactly the axis it is applied to?
- `column`: you want to keep the `V` axis with size 1. Which kind of index keeps its
  axis?

---

The key lines:

```
x / sum(x, over dim, keeping the axis)      # keepdim=True in PyTorch, keepdims=True in JAX and MLX

x[:, -1]                                    # (B, T, D) -> (B, D): an integer removes its axis

x[:, j : j + 1]                             # (B, V) -> (B, 1): a slice keeps its axis
```

`x[:, -1:].squeeze(1)` and `x[:, j][:, None]` are equally right. A `squeeze()` with no
argument is not.
