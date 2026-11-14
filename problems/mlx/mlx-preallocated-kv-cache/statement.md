Implement the KV cache that `mlx-lm` uses: buffers that grow 256 slots at a time,
filled by slice assignment, with only the filled part handed to attention. In MLX
the reason for this design is not the one you would expect. Slice assignment does not
write in place, so the cache copies its buffer on every token either way. What the
steps buy is that the buffer keeps its *shape*, and with it MLX's allocator can reuse
the memory it has just freed.

## Specification

```python
class KVCache:
    def __init__(self, step=256): ...
    def update_and_fetch(self, keys, values): ...  # returns (keys, values)
```

A cache holds the keys and values of one attention layer. It has four attributes:

- `step`: the constructor argument, a positive int.
- `offset`: the number of tokens stored, a Python int, `0` for a new cache.
- `keys`, `values`: the **buffers**. Both are `None` for a new cache. After the first
  update they are arrays of shape `(B, H, C, D_k)` and `(B, H, C, D_v)`, where `C` is
  the **capacity**. Slots `0` to `offset - 1` along axis 2 hold the tokens stored so
  far, oldest first. The contents of slots `offset` to `C - 1` are unspecified.

**`update_and_fetch(keys, values)`** appends the keys and values of `T` new tokens,
`keys` of shape `(B, H, T, D_k)` and `values` of shape `(B, H, T, D_v)` with
$T \ge 1$, and returns everything stored so far.

1. **Growth.** If the cache has no buffers yet, or if $\text{offset} + T > C$, the
   buffers are replaced by buffers of capacity

   $$
   C' = \text{offset} + \left\lceil \frac{T}{\text{step}} \right\rceil \cdot \text{step},
   $$

   whose slots `0` to `offset - 1` hold what the old ones held. Otherwise the buffers
   **keep their shape**.
2. **Write.** The new keys and values are stored in slots `offset` to
   `offset + T - 1`, and `offset` becomes `offset + T`.
3. **Fetch.** The result is the tuple `(keys, values)` of the first `offset` slots of
   each buffer: shapes `(B, H, offset, D_k)` and `(B, H, offset, D_v)`. The unused
   slots are never returned.

The buffers have the dtype of the arrays of the first update: a float16 cache stays
float16. `B`, `H`, `D_k`, `D_v`, and the dtype are the same in every update of one
cache.

`update_and_fetch` **evaluates nothing**: no `mx.eval`, no `.item()`, no conversion to
numpy. It is called once per layer in the middle of a forward pass, and the result
is evaluated with the rest of that pass.

![Buffer length as tokens are appended one at a time. With steps of 256 the buffers take three shapes on the way to 768 tokens; with concatenation they take a new shape at every token.](assets/growth.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `keys` (argument) | `(B, H, T, D_k)` | float32 or float16 |
| `values` (argument) | `(B, H, T, D_v)` | as `keys` |
| `cache.keys`, `cache.values` (buffers) | `(B, H, C, D_k)`, `(B, H, C, D_v)` | as the arguments |
| `keys`, `values` (result) | `(B, H, offset, D_k)`, `(B, H, offset, D_v)` | as the arguments |
| `cache.offset` | Python `int` | |

## Example

`step = 4`, one batch row, one head, `D_k = D_v = 1`. Token $i$ has key $i$ and
value $10 i$; the table shows axis 2 only.

```python
cache = KVCache(step=4)
```

| update | keys written | fits? | capacity after | `offset` after | keys returned |
|---|---|---|---|---|---|
| 1 | 1, 2, 3 | no buffer yet | $0 + \lceil 3/4 \rceil \cdot 4 = 4$ | 3 | 1, 2, 3 |
| 2 | 4 | $3 + 1 \le 4$ | 4 | 4 | 1, 2, 3, 4 |
| 3 | 5 | $4 + 1 > 4$ | $4 + 4 = 8$ | 5 | 1, 2, 3, 4, 5 |
| 4 | 6, ..., 11 | $5 + 6 > 8$ | $5 + \lceil 6/4 \rceil \cdot 4 = 13$ | 11 | 1, 2, ..., 11 |

The values returned are ten times the keys. After update 1 the buffer has four slots
and the result has three: the fourth slot exists and is not returned. Update 2 changes
no shape. In update 4 the capacity is counted from `offset` (5), not from the old
capacity (8): the three unused slots of the old buffer are dropped, and room for two
whole steps is added after the fifth token.

## Constraints

- $1 \le B \le 3$, $1 \le H \le 4$, head dimensions up to 64, `step` from 1 to 256, up
  to 1,000 tokens per cache.
- Returned keys and values, and the first `offset` slots of the buffers, are compared
  **exactly** with what was written.
- After every update the tests check that `offset` is a Python int with the right
  value and that both buffers have exactly the capacity rule 1 gives. The number of
  updates after which the buffers' shape differs from before is reported as
  `reallocations`; for $n$ tokens appended one at a time to a new cache it is
  $\lceil n / \text{step} \rceil$.
- Memory limit (on the Apple GPU). 600 updates of one token each, with `B = 1`,
  `H = 4`, `D_k = D_v = 64`, float32 (1 KiB per token and buffer), each followed by
  `mx.eval` of the result. Afterwards the memory MLX's allocator holds, in use
  (`mx.get_active_memory`) plus kept for reuse (`mx.get_cache_memory`), may be at
  most **24 MiB** above what it held before. A step-allocated cache holds 9 MiB (1.5
  MiB of buffers in use); a cache that changes shape on every update holds 70 MiB.

## Why this matters

`mx.get_cache_memory` is real memory: freed buffers that MLX keeps, up to a limit of
tens of gigabytes by default, because allocating from the system is slow and the next
array of the same size can take one over. A cache that grows by concatenation never
asks for the same size twice, so every buffer it frees is dead weight. At 4 KiB per
token and buffer, 1,024 generated tokens leave 2.9 GiB in the allocator to hold 8 MiB
of cache; with steps of 256 it is 52 MiB. That is the "MLX keeps eating memory while
generating" report, and it is why `mlx-lm` grows its cache in steps and calls
`mx.clear_cache()` every 256 tokens.
