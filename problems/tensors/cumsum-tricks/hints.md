Write each function as a loop over one row with a single running variable, and look
at what the variable holds at each step. For `exclusive_cumsum` it is the total so
far, read before the current entry is added. For the other two it is a count of how
many times something has been seen. Each of those variables, recorded at every step,
is a cumulative sum.

---

- `exclusive_cumsum`: `cumsum` gives the total after adding entry `i`. The total
  before adding it is the same list moved one place to the right, with a 0 in front
  and the last entry dropped.
- `first_true`: take the running count of true entries. Before the first true entry
  it is 0, and from there on it is at least 1. How many positions have a count of 0?
- `through_first_eos`: a position is kept when the number of `eos` tokens strictly
  before it is 0. "Strictly before" is what `exclusive_cumsum` computes, applied to
  the 0/1 tensor `ids == eos`.

---

The key lines, in PyTorch:

```
total = torch.cumsum(x, dim=-1)
torch.cat([torch.zeros_like(x[..., :1]), total[..., :-1]], dim=-1)   # exclusive_cumsum

(torch.cumsum(mask, dim=1) == 0).sum(dim=1)                          # first_true

is_eos = (ids == eos).to(torch.int64)
exclusive_cumsum(is_eos) == 0                                        # through_first_eos
```

JAX uses `jnp.concatenate` and `axis=`. MLX has `mx.cumsum(x, axis=-1,
inclusive=False)` for the first function.
