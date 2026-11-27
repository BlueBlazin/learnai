Compute the memory footprint of a decoder-only transformer's key-value (KV) cache from its configuration. Implement one function for the bytes added by a single token, and a second function for the total bytes of a batch at a given context length with optional sliding-window layers.

## Specification

```python
def kv_bytes_per_token(cfg: dict, dtype: str) -> int: ...
def kv_bytes(cfg: dict, dtype: str, batch: int, context: int,
             window_layers: dict[int, int]) -> int: ...
```

`cfg` is a dictionary describing the model architecture. Both functions read only these three keys and must ignore any others:

| key | meaning |
|---|---|
| `n_layers` | number of transformer blocks $L$ |
| `n_kv_heads` | number of key and value heads $H_{kv}$ |
| `head_dim` | width of a single head $h$ |

`dtype` names the element type stored in the cache. Each element occupies `itemsize` bytes:

| `dtype` | `itemsize` (bytes) |
|---|---|
| `"float32"` | 4 |
| `"float16"` | 2 |
| `"bfloat16"` | 2 |
| `"int8"` | 1 |

**What is cached.** For every token in every sequence, each transformer layer stores one key vector and one value vector of shape `(n_kv_heads, head_dim)`, which are the outputs of that layer's key and value projections. Across `batch` sequences of length `context`, one layer therefore stores two tensors (keys and values), each of shape `(batch, n_kv_heads, context, head_dim)`. Under grouped-query attention (GQA), multiple query heads share the `n_kv_heads` key and value heads, so only the `n_kv_heads` projections are stored. Queries are never cached.

**`kv_bytes_per_token(cfg, dtype)`** returns the number of bytes that one token of one sequence occupies when every layer caches it:

$$
\text{bytes per token} = 2 \cdot L \cdot H_{kv} \cdot h \cdot \text{itemsize}.
$$

**`kv_bytes(cfg, dtype, batch, context, window_layers)`** returns the total cache size in bytes for `batch` sequences that each hold `context` tokens.

- `window_layers` maps a layer index (from `0` to `n_layers - 1`) to that layer's sliding-window size $w$ in tokens. A sliding-window layer attends only to the most recent $w$ tokens (including the current token), so it stores `min(context, w)` tokens per sequence.
- Any layer index not present in `window_layers` is a global layer and stores all `context` tokens. `window_layers` may be empty.
- Letting $c_\ell$ be the number of tokens stored by layer $\ell$, the total cache size is:

$$
\text{bytes} = \text{batch} \cdot 2 \cdot H_{kv} \cdot h \cdot \text{itemsize} \cdot \sum_{\ell=0}^{L-1} c_\ell .
$$

Both functions must return a Python `int` and count tensor data only, with no allocator padding, block rounding, or per-sequence metadata. Neither function may modify its arguments.

## Example

Granite 4.2 3B (`ibm-granite/granite-4.2-3b`) has 40 layers and 40 query heads sharing 8 KV heads of width 64. It has no sliding-window layers.

```python
granite_3b = {"n_layers": 40, "n_heads": 40, "n_kv_heads": 8, "head_dim": 64}

kv_bytes_per_token(granite_3b, "bfloat16")       # 81920      = 2 * 40 * 8 * 64 * 2
kv_bytes_per_token(granite_3b, "float32")        # 163840
kv_bytes(granite_3b, "bfloat16", 1, 131072, {})  # 10737418240   one sequence, full context
kv_bytes(granite_3b, "bfloat16", 8, 4096, {})    # 2684354560    eight sequences of 4096
```

Each token takes 80 KiB. One sequence at the model's full 131,072-token context therefore requires 10 GiB of cache, which is more than the 7.3 GB of BF16 weights. Using the 40 query heads instead of the 8 KV heads would overestimate this as 50 GiB.

Gemma 3 4B (`google/gemma-3-4b-it`) has 34 layers, 4 KV heads of width 256, and a repeating pattern of five sliding-window layers (window size 1024) followed by one global layer. Layers 5, 11, 17, 23, and 29 are global, and the other 29 layers are windowed.

```python
gemma3_4b = {"n_layers": 34, "n_heads": 8, "n_kv_heads": 4, "head_dim": 256}
windows = {layer: 1024 for layer in range(34) if (layer + 1) % 6 != 0}

kv_bytes_per_token(gemma3_4b, "bfloat16")           # 139264
kv_bytes(gemma3_4b, "bfloat16", 1, 512, windows)    # 71303168    below the window: all 34 layers hold 512
kv_bytes(gemma3_4b, "bfloat16", 1, 8192, windows)   # 289406976   5 * 8192 + 29 * 1024 = 70656 layer-tokens
kv_bytes(gemma3_4b, "bfloat16", 1, 8192, {})        # 1140850688  if no layer were windowed
```

At 8192 tokens, the 29 windowed layers have stopped growing. The resulting cache is roughly one-quarter of what `context * kv_bytes_per_token` predicts.

## Constraints

- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= 256`, and `1 <= head_dim <= 1024`.
- `1 <= batch <= 4096` and `0 <= context <= 2,000,000`. A `context` of `0` returns `0`.
- Every window size in `window_layers` is at least `1`, and every key is in `range(n_layers)`.
- `dtype` is always one of the four strings in the table above.
- Results are compared exactly and routinely exceed $2^{32}$.

## Why this matters

Model weights are a fixed memory cost, while the KV cache grows with traffic. To a first approximation, a serving engine's maximum concurrency is the memory left after loading the weights divided by `kv_bytes` for one sequence. Inference engines like vLLM run this exact calculation at startup to decide how many cache blocks to allocate.

Being off by the GQA factor either leaves most of your GPU idle or admits requests until memory runs out mid-generation. The same formula determines whether a long context fits on a laptop at all, which is why modern architectures invest heavily in fewer KV heads, sliding windows, and quantized caches.
