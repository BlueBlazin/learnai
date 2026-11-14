Compute the size of a model's KV cache from its configuration: the bytes one token
adds, and the bytes a whole batch occupies at a given context length. The formula is a
product of five numbers, and each one has a plausible wrong value (query heads for KV
heads, bits for bytes, one tensor where there are two).

## Specification

```python
def kv_bytes_per_token(cfg: dict, dtype: str) -> int: ...
def kv_bytes(cfg: dict, dtype: str, batch: int, context: int,
             window_layers: dict[int, int]) -> int: ...
```

**What is cached.** For every token of every sequence, each layer stores one key
tensor and one value tensor of shape `(n_kv_heads, head_dim)`: the outputs of that
layer's key and value projections. The cache of one layer is therefore two tensors of
shape `(batch, n_kv_heads, context, head_dim)`. With grouped-query attention the
`n_heads` query heads share `n_kv_heads` key/value heads, and only the latter are
stored. Queries are never cached.

**`cfg`** is a dict. The functions use three of its keys and must ignore any others:

| key | meaning |
|---|---|
| `n_layers` | number of transformer blocks $L$ |
| `n_kv_heads` | number of key/value heads $H_{kv}$ |
| `head_dim` | width of one head $h$ |

**`dtype`** names the element type of the cache. One element occupies:

| `dtype` | bytes |
|---|---|
| `"float32"` | 4 |
| `"float16"` | 2 |
| `"bfloat16"` | 2 |
| `"int8"` | 1 |

**`kv_bytes_per_token(cfg, dtype)`** returns the bytes one token of one sequence
occupies when every layer stores it:

$$
\text{bytes per token} = 2 \cdot L \cdot H_{kv} \cdot h \cdot \text{itemsize}.
$$

**`kv_bytes(cfg, dtype, batch, context, window_layers)`** returns the bytes of the
whole cache for `batch` sequences that each hold `context` tokens.

- `window_layers` maps a layer index (`0` to `n_layers - 1`) to that layer's
  sliding-window size $w$ in tokens. A sliding-window layer attends only to the most
  recent $w$ tokens, the current one included, so it stores `min(context, w)` tokens
  per sequence. A layer that is not a key of `window_layers` is a global layer and
  stores all `context` tokens. The dict may be empty.
- With $c_\ell$ the number of tokens layer $\ell$ stores:

$$
\text{bytes} = \text{batch} \cdot 2 \cdot H_{kv} \cdot h \cdot \text{itemsize} \cdot \sum_{\ell=0}^{L-1} c_\ell .
$$

Both functions return a Python int and count the tensor data only: no allocator
padding, no block rounding, no per-sequence metadata. They do not modify their
arguments.

## Example

Granite 4.2 3B (`ibm-granite/granite-4.2-3b`) has 40 layers and 40 query heads that
share 8 KV heads of width 64. It has no sliding-window layers.

```python
granite_3b = {"n_layers": 40, "n_heads": 40, "n_kv_heads": 8, "head_dim": 64}

kv_bytes_per_token(granite_3b, "bfloat16")       # 81920      = 2 * 40 * 8 * 64 * 2
kv_bytes_per_token(granite_3b, "float32")        # 163840
kv_bytes(granite_3b, "bfloat16", 1, 131072, {})  # 10737418240   one sequence, full context
kv_bytes(granite_3b, "bfloat16", 8, 4096, {})    # 2684354560    eight sequences of 4096
```

Each token costs 80 KiB, so one sequence at the model's full 131,072-token context
needs exactly 10 GiB of cache, more than the 7.3 GB of BF16 weights. Sizing with the
40 query heads instead of the 8 KV heads would predict 50 GiB.

Gemma 3 4B (`google/gemma-3-4b-it`) has 34 layers, 4 KV heads of width 256, and a
repeating pattern of five sliding-window layers (window 1024) followed by one global
layer, so layers 5, 11, 17, 23, and 29 are global and the other 29 are windowed.

```python
gemma3_4b = {"n_layers": 34, "n_heads": 8, "n_kv_heads": 4, "head_dim": 256}
windows = {layer: 1024 for layer in range(34) if (layer + 1) % 6 != 0}

kv_bytes_per_token(gemma3_4b, "bfloat16")           # 139264
kv_bytes(gemma3_4b, "bfloat16", 1, 512, windows)    # 71303168    below the window: all 34 layers hold 512
kv_bytes(gemma3_4b, "bfloat16", 1, 8192, windows)   # 289406976   5 * 8192 + 29 * 1024 = 70656 layer-tokens
kv_bytes(gemma3_4b, "bfloat16", 1, 8192, {})        # 1140850688  if no layer were windowed
```

At 8192 tokens the 29 windowed layers have stopped growing, and the cache is a
quarter of what `context * kv_bytes_per_token` predicts.

## Constraints

- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= 256`, `1 <= head_dim <= 1024`.
- `1 <= batch <= 4096` and `0 <= context <= 2,000,000`; a context of `0` gives `0`.
- Window sizes are at least `1`, and every key of `window_layers` is in
  `range(n_layers)`.
- `dtype` is always one of the four names in the table.
- Results are compared exactly. They exceed $2^{32}$ routinely.

## Why this matters

The weights are a fixed cost; the KV cache is the part of memory that grows with
traffic. A server's concurrency limit is, to a first approximation, the memory left
after the weights divided by `kv_bytes` of one sequence, which is how engines such as
vLLM decide how many cache blocks they can allocate at startup. Get the formula wrong
by the GQA factor and you either refuse load the hardware could take or admit
requests until the allocator fails mid-generation. The same number decides whether a
long context fits on a laptop at all, and it is the reason architectures spend effort
on fewer KV heads, sliding windows, and quantized caches.
