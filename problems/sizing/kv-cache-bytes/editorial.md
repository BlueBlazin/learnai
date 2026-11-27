## The idea

Attention at position $t$ needs the keys and values from every earlier position. Because past keys and values never change once computed, a decoder caches them instead of recomputing them at every step. The cache size follows directly from the shapes of the two tensors stored per layer:

```
keys:   (batch, n_kv_heads, tokens, head_dim)
values: (batch, n_kv_heads, tokens, head_dim)
```

For a model with $L$ layers, $H_{kv}$ key/value heads, head width $h$, and `itemsize` bytes per element, multiplying the tensor axes, the two tensors (K and V), the layer count, and the element size gives:

$$
\text{bytes} = \underbrace{2}_{K,\,V} \cdot \underbrace{L}_{\text{layers}} \cdot
\underbrace{H_{kv} \cdot h}_{\text{per token, per layer}} \cdot
\underbrace{\text{itemsize}}_{\text{bytes}} \cdot \text{batch} \cdot \text{context}.
$$

Everything except `batch` and `context` is fixed by the model checkpoint and the cache dtype. That is why "bytes per token" is a useful rule-of-thumb number to remember for a model, such as 80 KiB for Granite 4.2 3B in BF16 or 28 KiB for Granite 4.0 350M.

**Why KV heads and not query heads.** Grouped-query attention (Ainslie et al., 2023) lets multiple query heads share a single key head and a single value head. The query projection still produces `n_heads * head_dim` features, but queries are used once and discarded. Only the `n_kv_heads * head_dim` features of K and V are cached. Shrinking the KV cache is the main goal of GQA. For example, Granite 4.2 3B's 40-to-8 head grouping cuts cache memory by 5x without reducing attention width.

**Sliding windows.** A layer that attends only to the last $w$ tokens never reads older keys or values, so an engine can evict them with a ring buffer of $w$ slots. Such a layer stores `min(context, w)` tokens. When sliding windows are present, the total cache is a sum over layers rather than a simple product with `n_layers`. Once `context` exceeds the window size, cache growth depends only on the number of *global* layers (5 of 34 in Gemma 3 4B).

## Reference solution (Python)

```python
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}


def kv_bytes_per_token(cfg, dtype):
    return 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]


def kv_bytes(cfg, dtype, batch, context, window_layers):
    layer_tokens = sum(
        min(context, window_layers.get(layer, context)) for layer in range(cfg["n_layers"])
    )
    per_layer_token = 2 * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    return batch * layer_tokens * per_layer_token
```

When `window_layers` is empty, `kv_bytes` simplifies to `batch * context * kv_bytes_per_token(cfg, dtype)`.

## Where the numbers come from

The table below was read on 2026-11-14 from each repository's `config.json` on the Hugging Face Hub (`https://huggingface.co/<repo>/raw/main/config.json`).

| model | layers | heads / KV heads | head dim | bytes per token, BF16 |
|---|---|---|---|---|
| `ibm-granite/granite-4.0-350m` | 28 | 16 / 4 | 64 | 28,672 |
| `ibm-granite/granite-4.2-3b` | 40 | 40 / 8 | 64 | 81,920 |
| `Qwen/Qwen3-0.6B` | 28 | 16 / 8 | 128 | 114,688 |
| `google/gemma-3-4b-it` | 34 | 8 / 4 | 256 | 139,264 (global layers only: 20,480) |

Granite's config omits `head_dim`, which is `hidden_size / num_attention_heads`. Gemma 3 stores its values under `text_config` with `sliding_window: 1024` and `sliding_window_pattern: 6`. (The official Gemma 3 repo is gated, and ungated mirrors such as `unsloth/gemma-3-4b-it` carry the same config.) Comparing the two small models shows how architecture drives cache size. Qwen3-0.6B has 1.7x the parameters of Granite 4.0 350M (596 M vs. 352 M), yet 4x the KV cache per token.

## Common mistakes

- **Using query heads instead of KV heads.** Using `n_heads * head_dim` (or `d_model`, a pre-GQA habit) overestimates the cache by the grouping factor: 5x for Granite 4.2 3B and 4x for Granite 4.0 350M. A capacity plan built on this number admits only a fraction of the load the hardware can handle.
- **Forgetting that there are two tensors.** Each layer caches both keys *and* values. Dropping the factor of 2 produces plausible numbers that are off by half.
- **Charging windowed layers for the full context.** Ignoring `window_layers` looks right below the window size, then balloons past it (3.9x too large at 8192 tokens for Gemma 3 4B, and 6.5x at 131,072 tokens). Applying the window to every layer instead of only windowed layers is the symmetric bug.
- **Leaving out the batch axis.** The cache is allocated per sequence, so `batch` is the factor that scales with concurrent traffic.
- **Confusing bits with bytes.** `float16` is 16 bits, which is 2 bytes. Parsing the digits out of the dtype string makes the answer 8x too large. Likewise, `bfloat16` is a 16-bit (2-byte) format, not a 4-byte `float32`.

## Complexity

`kv_bytes` runs in $O(L)$ time (one term per layer) and $O(1)$ space. `kv_bytes_per_token` runs in $O(1)$ time and space.

## In the wild

- **Admission and concurrency.** Dividing free memory (after loading weights) by the KV cache of one sequence at your maximum supported context length gives the number of concurrent sequences a server can hold. vLLM performs this calculation at startup in blocks of 16 tokens and logs the result as its maximum concurrency.
- **Real allocations are slightly larger.** Paged caches round each sequence up to a whole block, engines preallocate pools, and some deployments store the KV cache in a different dtype than the weights. Because this problem counts tensor data only, treat its output as a lower bound.
- **Shrinking the cache.** Every cache-reduction technique targets one factor in the formula: fewer KV heads (GQA, and multi-query attention with a single KV head), fewer tokens per layer (sliding windows as in Gemma 3), smaller elements (8-bit cache quantization in `llama.cpp` and `mlx-lm`, which adds per-group scales omitted by `"int8"` here), and fewer cached layers (architectures that share KV caches across layers).
