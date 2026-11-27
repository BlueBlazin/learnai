Write each function as a loop over one row with a single running variable, and inspect what that variable holds at each step. For `exclusive_cumsum`, it is the running total before adding the current entry. For the other two functions, it is a running count of how many times a condition has been met. Recording that variable at every step is a cumulative sum.

---

- **`exclusive_cumsum`**: `cumsum` gives the total after adding entry `i`. The total before adding it is the same sequence shifted one position to the right, with a `0` prepended and the last entry dropped.
- **`first_true`**: Take the running count of `True` entries. Before the first `True` entry the count is `0`, and from that position onward it is at least `1`. Count how many positions have a running count of `0`.
- **`through_first_eos`**: A position is kept when the number of `eos` tokens strictly before it is `0`. Apply `exclusive_cumsum` to the 0/1 indicator `ids == eos` and compare to `0`.

---

The key lines in PyTorch:

```python
total = torch.cumsum(x, dim=-1)
torch.cat([torch.zeros_like(x[..., :1]), total[..., :-1]], dim=-1)   # exclusive_cumsum

(torch.cumsum(mask, dim=1) == 0).sum(dim=1)                          # first_true

is_eos = (ids == eos).to(torch.int64)
exclusive_cumsum(is_eos) == 0                                        # through_first_eos
```

JAX uses `jnp.concatenate` and `axis=`. MLX supports `mx.cumsum(x, axis=-1, inclusive=False)` directly for `exclusive_cumsum`.
