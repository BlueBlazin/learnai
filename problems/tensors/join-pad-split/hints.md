For each function, write the shapes of the pieces and of the result. If the result
has an axis that none of the pieces has, something must create it. If it does not,
the pieces are joined along an axis they share, and the only question is the order.

---

- `pad_rows`: first make every row the same length, then join the rows. Each row
  needs `T_max - n_i` copies of `pad` on one side, which is a `cat` of two
  one-dimensional tensors or one `pad` call. The padded rows are `B` tensors of
  shape `(T_max,)`, and the result is `(B, T_max)`: a new axis. The mask follows
  from the lengths alone, as in Masks from `arange`; do not derive it from `ids`.
- `shift_right`: the result is a column of `bos`, shape `(B, 1)`, followed by all
  columns of `ids` except the last, shape `(B, T - 1)`. Both pieces have the column
  axis.
- `stack_steps`: `T` tensors of shape `(B, V)` become `(B, T, V)`. The new axis is
  in the middle.

---

The key lines, in PyTorch:

```
F.pad(row, (gap, 0), value=pad)                # left; (0, gap) for right
ids = torch.stack(padded)                      # (B, T_max)
position = torch.arange(width)[None, :]
mask = position >= (width - lengths)[:, None]  # left; position < lengths[:, None] for right

first = torch.full_like(ids[:, :1], bos)
torch.cat([first, ids[:, :-1]], dim=1)         # shift_right

torch.stack(steps, dim=1)                      # stack_steps
```

In JAX: `jnp.pad(row, (gap, 0), constant_values=pad)`, `jnp.stack`,
`jnp.concatenate(..., axis=1)`. MLX has the same functions under `mx`.
