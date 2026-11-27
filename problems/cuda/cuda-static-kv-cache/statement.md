Implement `StaticCache`, a fixed-shape key-value (KV) cache for a GPU decode loop. Every buffer is allocated once at full length, every write updates that memory in place using a device-side position tensor, and a boolean attention mask hides unused slots.

## Specification

```python
class StaticCache:
    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, device, dtype): ...
    def update(self, layer, k, v, pos): ...  # returns (k_buffer, v_buffer)
    def mask(self, pos, n_new): ...  # returns a bool tensor
    def reset(self): ...
```

**`__init__`** allocates one zero-filled key buffer and one zero-filled value buffer for each of the `n_layers` layers on `device` with dtype `dtype`. Expose them as public lists `cache.k` and `cache.v` of length `n_layers`. Slot `j` along axis 2 (the sequence axis) holds the token at sequence position `j`.

**`update(layer, k, v, pos)`** writes `T` new tokens into layer `layer`.

- `k` and `v` hold the new keys and values on `device` with dtype `dtype`.
- `pos` is a 0-D `int64` tensor on `device` giving the starting slot of the first new token. The caller guarantees $0 \le \text{pos}$ and $\text{pos} + T \le \text{max\_len}$.
- Write `k` into slots `pos` through `pos + T - 1` of `cache.k[layer]` and `v` into the same slots of `cache.v[layer]`, leaving all other slots unchanged.
- Return `(cache.k[layer], cache.v[layer])` as full buffers of length `max_len`, regardless of `pos` and `T`.

**`mask(pos, n_new)`** returns a boolean attention mask on `device` for `n_new` tokens (a Python `int`) written starting at `pos`. Row $i$ corresponds to the query token in slot $\text{pos} + i$, which may attend to itself and every earlier cache slot $j$:

$$
\text{mask}[i, j] = (j \le \text{pos} + i).
$$

The model applies this mask to attention scores over all `max_len` slots, setting scores where the mask is `False` to $-\infty$ before softmax.

![A cache with `max_len = 8` after two tokens are written at `pos=3`, and the two mask rows. Only the mask keeps the unused slots out of attention.](assets/static-cache-mask.svg)

**`reset()`** prepares the cache for the next request by filling every key and value buffer with zeros in place and returning `None`.

Four rules keep the cache static, and the judge checks all four:

1. **Shapes never change.** `update` and `mask` always return tensors with the exact shapes in the table below.
2. **Memory never changes.** `update` and `reset` write in place into the tensors created in `__init__`. Every buffer keeps the same `data_ptr()` across all calls, so an earlier reference to a buffer reflects all subsequent writes.
3. **No host reads.** No method converts `pos` or any other tensor into a Python number. Do not call `int(pos)` or `pos.item()`, and do not use `pos` as a slice bound.
4. **All allocations happen on `device`.** Every tensor factory call (`torch.zeros`, `torch.arange`, and so on) must pass `device=`.

> [!WARNING]
> Slots at and beyond the current sequence length are not empty. They contain zeros after a reset, or leftover values from an earlier request. The judge fills unused slots with large finite numbers, and attention using `mask` must match what it would produce if those slots did not exist.

## Shapes

| Name | Shape | Dtype |
|---|---|---|
| `cache.k[layer]`, `cache.v[layer]` | `(batch, n_kv_heads, max_len, head_dim)` | `dtype` |
| `k`, `v` passed to `update` | `(batch, n_kv_heads, T, head_dim)` | `dtype` |
| `pos` | `()` | `int64` |
| result of `update` | two tensors `(batch, n_kv_heads, max_len, head_dim)` | `dtype` |
| result of `mask` | `(n_new, max_len)` | `bool` |

## Example

Consider 1 layer, batch size 1, 1 head, `max_len = 4`, `head_dim = 2`, and the three tokens from `preallocated-kv-cache`:

```python
k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # shape (1, 1, 3, 2) once batched
v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

cache = StaticCache(1, 1, 1, 4, 2, device, torch.float32)

k_buf, v_buf = cache.update(0, k[:, :, :2], v[:, :, :2], torch.tensor(0))
# k_buf = [[[[1, 0], [0, 1], [0, 0], [0, 0]]]]            shape (1, 1, 4, 2)
cache.mask(torch.tensor(0), 2)
# [[True, False, False, False],
#  [True, True,  False, False]]

k_buf2, v_buf2 = cache.update(0, k[:, :, 2:], v[:, :, 2:], torch.tensor(2))
# k_buf2 = [[[[1, 0], [0, 1], [1, 1], [0, 0]]]]           shape (1, 1, 4, 2) again
cache.mask(torch.tensor(2), 1)
# [[True, True, True, False]]
```

`k_buf2` points to the same memory as `k_buf`, so `k_buf` now reflects the third token as well. For the third token's query $[1, 1]$, the scaled dot-product scores against the four slots are $[0.7071, 0.7071, 1.4142, 0]$. The mask sets the last score to $-\infty$, giving weights $[0.2483, 0.2483, 0.5035, 0]$ and output $[3.5105, 4.5105]$, which matches a 3-token cache. Without the mask, the zero key in slot 3 receives unnormalized weight $e^0 = 1$, shifting the output to $[3.1276, 4.0186]$.

## Constraints

- $1 \le \text{n\_layers} \le 4$, $1 \le \text{batch} \le 3$, $1 \le \text{n\_kv\_heads} \le 4$, $4 \le \text{max\_len} \le 16$, $2 \le \text{head\_dim} \le 8$ in the CPU cases. `dtype` is `float32`, `float16`, or `bfloat16`.
- Each request writes blocks of 1 to 7 tokens at consecutive positions starting at 0. After `reset()`, the next request starts at 0 again.
- Buffer contents and masks are compared exactly. Attention over the full buffers using your mask is compared against attention over a concatenation-grown cache with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- Unused slots always hold finite numbers (a `NaN` in an unused value slot would propagate through $0 \cdot \text{NaN}$ regardless of the mask).
- No Python loop over tokens, heads, or batch rows is needed.
- **What runs where.** Cases without a hardware mark run on the CPU. They monitor factory calls and host reads in your file, compare `data_ptr()` across calls, and run one check on `torch.device("meta")` (where tensors have shapes but no data, so any read of `pos` raises). Two cases run only on a machine with an NVIDIA GPU: a CUDA request in `float32` and `float16`, and 256 decode steps during which `torch.cuda.memory_reserved()` may grow by at most 2 MiB. The GPU cases were written from the PyTorch 2.15 documentation and could not be run when the problem was authored.

## Why this matters

A KV cache that grows with `torch.cat` allocates two new tensors per layer on every token and copies all prior keys and values into them. That causes quadratic copying, allocator churn, and a changing tensor shape at every step.

Changing shapes block the two main optimizations for GPU decoding. `torch.compile` specializes on tensor shapes and recompiles or falls back when shapes vary, while a CUDA graph replays recorded kernels on fixed memory addresses. A static cache satisfies both as long as writes stay in place (rebinding a buffer changes its address) and the write position stays on the device (calling `int(pos)` stalls the GPU pipeline on every token and fails during CUDA graph capture). The tradeoff is that correctness depends on the mask. An off-by-one error lets the model attend to an unwritten slot or to a previous request.
