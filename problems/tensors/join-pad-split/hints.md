For each function, write down the shapes of the input pieces and the shape of the result. If the result has an axis that none of the pieces has, you need an operation that creates a new axis (`stack`). If all axes already exist in the pieces, you join along a shared axis (`cat` or `pad`) and only need to pick the order.

---

- `pad_rows`: Pad every row to length `T_max` first, then join the padded rows into a batch. Each row needs `T_max - n_i` copies of `pad` on one side, which is either a `cat` of two 1D tensors or a single `pad` call. The padded rows are `B` tensors of shape `(T_max,)`, and the output `(B, T_max)` adds a new batch axis. Build the mask from the row lengths and `arange(T_max)`, not by comparing `ids` with `pad`.
- `shift_right`: Join a `(B, 1)` column filled with `bos` and all columns of `ids` except the last, which has shape `(B, T - 1)`. Both pieces already have the column axis.
- `stack_steps`: You have `T` tensors of shape `(B, V)` and want shape `(B, T, V)`. Insert the new axis at position 1.

---

The key lines in PyTorch:

```
F.pad(row, (gap, 0), value=pad)                # left; (0, gap) for right
ids = torch.stack(padded)                      # (B, T_max)
position = torch.arange(width)[None, :]
mask = position >= (width - lengths)[:, None]  # left; position < lengths[:, None] for right

first = torch.full_like(ids[:, :1], bos)
torch.cat([first, ids[:, :-1]], dim=1)         # shift_right

torch.stack(steps, dim=1)                      # stack_steps
```

In JAX, use `jnp.pad(row, (gap, 0), constant_values=pad)`, `jnp.stack`, and `jnp.concatenate(..., axis=1)`. MLX provides the same functions under `mx`.
