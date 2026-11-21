Implement a KV cache for a GPU decode loop in which nothing changes from one step to
the next except numbers: every buffer is allocated once at full length, every write
goes into that memory in place, and the position of the write is itself a tensor on
the device. The model then sees tensors of the same shape, at the same address, at
every step, and a mask has to stand in for the length the shapes no longer carry.

## Specification

```python
class StaticCache:
    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, device, dtype): ...
    def update(self, layer, k, v, pos): ...  # returns (k_buffer, v_buffer)
    def mask(self, pos, n_new): ...  # returns a bool tensor
    def reset(self): ...
```

**The constructor** allocates, for each of the `n_layers` layers, one key buffer and
one value buffer of shape `(batch, n_kv_heads, max_len, head_dim)`, zero-filled, of
dtype `dtype`, on `device`. They are public: `cache.k` and `cache.v` are lists of
`n_layers` tensors. Slot `j` along axis 2 belongs to the token at position `j`.

**`update(layer, k, v, pos)`** stores the keys and values of `T` new tokens.

- `k`, `v`: tensors of shape `(batch, n_kv_heads, T, head_dim)`, of dtype `dtype`, on
  `device`.
- `pos`: a **0-d int64 tensor on `device`**: the slot of the first new token. The
  caller guarantees $0 \le \text{pos}$ and $\text{pos} + T \le \text{max\_len}$.
- Writes `k` into slots `pos` to `pos + T - 1` of `cache.k[layer]` and `v` into the
  same slots of `cache.v[layer]`, and changes no other slot.
- Returns the pair `(cache.k[layer], cache.v[layer])`: the full buffers, of shape
  `(batch, n_kv_heads, max_len, head_dim)` whatever `pos` and `T` are.

**`mask(pos, n_new)`** returns the attention mask for the `n_new` tokens written at
`pos` (`n_new` is a Python int, `pos` as above): a bool tensor of shape
`(n_new, max_len)` on `device` with

$$
\text{mask}[i, j] = (j \le \text{pos} + i).
$$

Row $i$ belongs to the query of the token in slot $\text{pos} + i$, which may attend
to itself and to every earlier slot. The model applies it to the attention scores
over all `max_len` slots, setting the scores where the mask is `False` to $-\infty$
before the softmax.

![A cache with `max_len = 8` after two tokens are written at `pos=3`, and the two mask rows. Only the mask keeps the unused slots out of attention.](assets/static-cache-mask.svg)

**`reset()`** prepares the cache for the next request: it fills every buffer with
zeros and returns `None`.

Four rules make the cache static, and the judge checks each of them:

1. **Shapes never change.** The tensors returned by `update` and `mask` have the
   shapes above at every call.
2. **Memory never changes.** The tensors allocated by the constructor stay the
   buffers for the lifetime of the cache: `update` and `reset` write into them in
   place. `cache.k[layer].data_ptr()` is the same before and after every call, and a
   reference to a buffer taken earlier shows every later write.
3. **Nothing is read on the host.** No method converts `pos` or any other tensor to
   a Python number: no `int(pos)`, `pos.item()`, or `pos` as a slice bound.
4. **Everything is created on `device`.** Every factory call (`torch.zeros`,
   `torch.arange`, ...) passes `device=`.

> [!WARNING]
> The slots at and beyond the current length are not empty. They hold zeros after a
> reset, and otherwise whatever an earlier request left there; the judge fills them
> with large finite numbers. Attention through `mask` must be exactly what it would
> be if those slots did not exist.

## Shapes

| name | shape | dtype |
|---|---|---|
| `cache.k[layer]`, `cache.v[layer]` | `(batch, n_kv_heads, max_len, head_dim)` | `dtype` |
| `k`, `v` passed to `update` | `(batch, n_kv_heads, T, head_dim)` | `dtype` |
| `pos` | `()` | int64 |
| result of `update` | two tensors `(batch, n_kv_heads, max_len, head_dim)` | `dtype` |
| result of `mask` | `(n_new, max_len)` | bool |

## Example

One layer, one batch row, one head, `head_dim = 2`, four slots, and the three tokens
of `preallocated-kv-cache`:

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

`k_buf2` is the same memory as `k_buf`, and `k_buf`, obtained before the second call,
now shows the third token as well. With the query $[1, 1]$ for the third token, the
scores against the four slots are $[0.7071, 0.7071, 1.4142, 0]$; the mask turns the
last one into $-\infty$, the weights are $[0.2483, 0.2483, 0.5035, 0]$, and the
output is $[3.5105, 4.5105]$, the same as with a cache that holds three tokens.
Without the mask, the zero key in slot 3 gets the weight $e^0$ and the output is
$[3.1276, 4.0186]$.

## Constraints

- $1 \le \text{n\_layers} \le 4$, $1 \le \text{batch} \le 3$, $1 \le \text{n\_kv\_heads} \le 4$,
  $4 \le \text{max\_len} \le 16$, $2 \le \text{head\_dim} \le 8$ in the CPU cases;
  `dtype` is float32, float16, or bfloat16.
- A request writes blocks of 1 to 7 tokens at consecutive positions starting at 0;
  after `reset()` the next request starts at 0 again.
- Buffer contents and masks are compared exactly. Attention over the full buffers
  with your mask is compared with attention over a cache grown by concatenation, to
  a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- Unused slots hold finite numbers (a NaN in an unused value slot would reach the
  output through $0 \cdot \text{NaN}$ whatever the mask says).
- No Python loop over tokens, heads, or batch rows is needed.
- **What runs where.** The cases without a hardware mark run on the CPU. They watch
  the factory functions and the host reads made by your file, compare `data_ptr()`
  across calls, and one of them runs the cache on `torch.device("meta")`, whose
  tensors have a shape and no data, so that any read of `pos` raises. Two cases run
  only on a machine with an NVIDIA GPU: a request on the CUDA device in float32 and
  float16, and 256 decode steps during which `torch.cuda.memory_reserved()` may grow
  by at most 2 MiB. The GPU cases were written from the PyTorch 2.15 documentation
  and could not be run when the problem was authored.

## Why this matters

A cache that grows with `torch.cat` allocates two new tensors per layer per token and
copies everything cached so far into them: quadratic copying, a stream of
differently sized blocks through the allocator, and a new shape at every step. The
new shape is what rules out the two big optimizations of a GPU decode loop.
`torch.compile` specializes on shapes and recompiles or falls back when they move,
and a CUDA graph replays recorded kernels on recorded addresses, so it needs the same
memory every time. A static cache gives both what they need, provided the writes are
in place (a rebound buffer is a new address) and the position never comes back to
the host (an `int(pos)` stalls the pipeline once per token, and inside a captured
graph it is an error). The price is the mask: get it wrong by one slot and the model
attends to a token that was never written, or to the previous request's.
