Implement `KVCache`, the step-allocated key-value cache used by `mlx-lm`. The cache grows its buffers in fixed steps (256 tokens by default), writes new tokens with slice assignment, and returns only the filled prefix to attention.

In MLX, slice assignment does not mutate memory in place, so the cache still copies its buffers on every token. Growing in fixed steps keeps the buffer shape constant between reallocations, which lets MLX's memory allocator reuse freed buffers instead of accumulating unused allocations.

## Specification

```python
class KVCache:
    def __init__(self, step=256): ...
    def update_and_fetch(self, keys, values): ...  # returns (keys, values)
```

A `KVCache` instance stores the keys and values for one attention layer across batch size `B`, attention heads `H`, key head dimension `D_k`, and value head dimension `D_v`. It maintains four attributes:

- `step`: a positive Python `int` giving the allocation step size.
- `offset`: a Python `int` counting how many tokens are currently stored (`0` for a new cache).
- `keys`, `values`: the internal buffers. Both start as `None`. After the first update, they have shapes `(B, H, C, D_k)` and `(B, H, C, D_v)`, where `C` is the current token capacity along axis 2. Slots `0` through `offset - 1` hold the stored tokens in chronological order (oldest first). Slots `offset` through `C - 1` hold unspecified padding.

`update_and_fetch(keys, values)` appends `T` new tokens (with $T \ge 1$) of shapes `(B, H, T, D_k)` and `(B, H, T, D_v)` and returns all tokens stored so far. Each call performs three steps:

1. **Growth.** If the buffers are still `None`, or if $\text{offset} + T > C$, replace them with buffers of capacity

   $$
   C' = \text{offset} + \left\lceil \frac{T}{\text{step}} \right\rceil \cdot \text{step},
   $$

   preserving slots `0` through `offset - 1` from the previous buffers. Otherwise, keep the existing buffer shapes unchanged.
2. **Write.** Store the incoming `keys` and `values` into slots `offset` through `offset + T - 1` along axis 2, then set `offset` to `offset + T`.
3. **Fetch.** Return the tuple `(keys, values)` sliced to the first `offset` slots along axis 2, with shapes `(B, H, offset, D_k)` and `(B, H, offset, D_v)`. Never return the unused tail slots.

The buffers must preserve the dtype of the arrays from the first update (for example, a float16 cache stays float16). Across all updates to a single cache, `B`, `H`, `D_k`, `D_v`, and the dtype remain constant.

`update_and_fetch` must stay completely lazy and evaluate nothing. Do not call `mx.eval`, `.item()`, or convert arrays to NumPy. The cache runs once per layer inside a forward pass, and the caller evaluates the graph at the end of the pass.

![Buffer length as tokens are appended one at a time. Steps of 256 give three shapes on the way to 768 tokens, where concatenation gives 768.](assets/growth.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `keys` (argument) | `(B, H, T, D_k)` | float32 or float16 |
| `values` (argument) | `(B, H, T, D_v)` | same as `keys` |
| `cache.keys`, `cache.values` (buffers) | `(B, H, C, D_k)`, `(B, H, C, D_v)` | same as arguments |
| `keys`, `values` (result) | `(B, H, offset, D_k)`, `(B, H, offset, D_v)` | same as arguments |
| `cache.offset` | Python `int` | |

## Example

Consider a cache with `step = 4`, `B = 1`, `H = 1`, and `D_k = D_v = 1`. Suppose token $i$ has key $i$ and value $10i$. The table below shows only axis 2.

```python
cache = KVCache(step=4)
```

| update | keys written | fits? | capacity after | `offset` after | keys returned |
|---|---|---|---|---|---|
| 1 | 1, 2, 3 | no buffer yet | $0 + \lceil 3/4 \rceil \cdot 4 = 4$ | 3 | 1, 2, 3 |
| 2 | 4 | $3 + 1 \le 4$ | 4 | 4 | 1, 2, 3, 4 |
| 3 | 5 | $4 + 1 > 4$ | $4 + 4 = 8$ | 5 | 1, 2, 3, 4, 5 |
| 4 | 6, ..., 11 | $5 + 6 > 8$ | $5 + \lceil 6/4 \rceil \cdot 4 = 13$ | 11 | 1, 2, ..., 11 |

Each returned `values` array is ten times the corresponding `keys` array. After update 1, the buffer holds four slots, while the returned slice contains only the first three. Update 2 fits in the remaining slot without changing the buffer shape. In update 4, the new capacity is computed from `offset` (5) rather than the old capacity (8). The three unused tail slots are discarded, and two full steps of 4 slots are appended after the fifth token.

## Constraints

- $1 \le B \le 3$, $1 \le H \le 4$, `D_k` and `D_v` up to 64, `step` from 1 to 256, and up to 1,000 tokens per cache.
- Returned keys and values, as well as the first `offset` slots of the buffers, must match the written values **exactly**.
- After every update, `offset` must be a Python `int` with the exact token count, and both buffers must have the exact capacity specified by rule 1. The test suite tracks `reallocations`, the number of updates that change the buffer shape. Appending $n$ tokens one at a time to a new cache must produce $\lceil n / \text{step} \rceil$ reallocations.
- **Memory limit (Apple GPU).** The test runs 600 single-token updates with `B = 1`, `H = 4`, `D_k = D_v = 64`, and float32 (1 KiB per token per buffer), calling `mx.eval` on the result after each update. Afterward, total allocator memory, meaning active (`mx.get_active_memory`) plus cached (`mx.get_cache_memory`), must exceed the starting memory by at most **24 MiB**. A step-allocated cache uses about 9 MiB (1.5 MiB active), whereas a cache that changes shape on every update uses 70 MiB.

## Why this matters

`mx.get_cache_memory` tracks real physical memory that MLX keeps after freeing buffers so it can satisfy future allocations of the same size without asking the operating system. By default, this cache can grow to tens of gigabytes. A KV cache that appends via `mx.concatenate` on every token requests a brand-new buffer size on every step, so previously freed buffers are never reused and sit idle in the allocator cache.

At 4 KiB per token per buffer, generating 1,024 tokens with per-step concatenation leaves 2.9 GiB trapped in the allocator just to hold 8 MiB of live cache. Growing in steps of 256 reduces that footprint to 52 MiB. This allocator behavior explains why naive MLX generation loops appear to leak memory, and why `mlx-lm` both allocates in 256-token steps and calls `mx.clear_cache()` every 256 tokens.
