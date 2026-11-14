## The idea

Attention at position $t$ needs the keys and values of every earlier position. They
do not change once computed, so a decoder stores them instead of recomputing them:
that is the KV cache. Its size follows from the shape of what is stored. Per layer:

```
keys:   (batch, n_kv_heads, tokens, head_dim)
values: (batch, n_kv_heads, tokens, head_dim)
```

Multiply the axes, the two tensors, the layers, and the bytes per element:

$$
\text{bytes} = \underbrace{2}_{K,\,V} \cdot \underbrace{L}_{\text{layers}} \cdot
\underbrace{H_{kv} \cdot h}_{\text{per token, per layer}} \cdot
\underbrace{\text{itemsize}}_{\text{bytes}} \cdot \text{batch} \cdot \text{context}.
$$

Everything except `batch` and `context` is fixed by the checkpoint and the cache
dtype, which is why "bytes per token" is the number to remember for a model: 80 KiB
for Granite 4.2 3B in BF16, 28 KiB for Granite 4.0 350M.

**Why KV heads and not heads.** Grouped-query attention (Ainslie et al., 2023) lets
several query heads share one key head and one value head. The query projection still
produces `n_heads * head_dim` features, but queries are used once and thrown away;
only the `n_kv_heads * head_dim` features of K and V are kept. Shrinking the cache is
the purpose of the technique: Granite 4.2 3B's 40-over-8 grouping cuts it by a factor
of five at no cost in attention width.

**Sliding windows.** A layer that only attends to the last $w$ tokens never reads
older keys, so an engine can drop them (a ring buffer of $w$ slots). Such a layer
holds `min(context, w)` tokens. The model's cache is then a sum over layers, not a
product with `n_layers`, and it grows with the number of *global* layers once the
context passes the window: for Gemma 3 4B, 5 of 34.

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

With an empty `window_layers`, `kv_bytes` is `batch * context * kv_bytes_per_token`.

## Where the numbers come from

Read on 2026-11-14 from each repository's `config.json` on the Hugging Face Hub
(`https://huggingface.co/<repo>/raw/main/config.json`).

| model | layers | heads / KV heads | head dim | bytes per token, BF16 |
|---|---|---|---|---|
| `ibm-granite/granite-4.0-350m` | 28 | 16 / 4 | 64 | 28,672 |
| `ibm-granite/granite-4.2-3b` | 40 | 40 / 8 | 64 | 81,920 |
| `Qwen/Qwen3-0.6B` | 28 | 16 / 8 | 128 | 114,688 |
| `google/gemma-3-4b-it` | 34 | 8 / 4 | 256 | 139,264 (global layers only: 20,480) |

Granite's config has no `head_dim` key; it is `hidden_size / num_attention_heads`.
The Gemma 3 values are in `text_config` (`sliding_window: 1024`,
`sliding_window_pattern: 6`; the repository is gated, and ungated mirrors such as
`unsloth/gemma-3-4b-it` carry the same file). Note the comparison between the two
small models: Qwen3-0.6B has 1.7 times the parameters of Granite 4.0 350M (596 M
against 352 M) and four times the cache per token.

## Common mistakes

- **Query heads in place of KV heads.** `n_heads * head_dim` (or `d_model`, the
  pre-GQA habit) overestimates by the grouping factor: 5x for Granite 4.2 3B, 4x for
  Granite 4.0 350M. A capacity plan built on it admits a fifth of the load the
  hardware can carry.
- **Forgetting that there are two tensors.** Keys *and* values. Without the factor of
  2 everything looks plausible and is half the truth.
- **Charging windowed layers for the full context.** Correct below the window, then
  too large: 3.9x at 8192 tokens for Gemma 3 4B, 6.5x at 131,072. The reverse error
  (applying the window to every layer) is just as easy.
- **Leaving out the batch axis.** The cache is per sequence. This is the term that
  scales with traffic, so it is the one a capacity plan exists to compute.
- **Bits for bytes.** `float16` is 16 bits and 2 bytes. Parsing the digits out of the
  dtype name gives an answer 8x too large (and `bfloat16` is not 4 bytes because it
  resembles a float32: it is a 16-bit format).

## Complexity

$O(L)$ for `kv_bytes` (one term per layer), $O(1)$ for `kv_bytes_per_token`.

## In the wild

- **Admission and concurrency.** Free memory after the weights, divided by the cache
  of one sequence at the longest context you promise, is the number of sequences you
  can hold. vLLM does this arithmetic at startup, in blocks of 16 tokens, and reports
  the result as its maximum concurrency.
- **Real allocations are a little larger.** Paged caches round each sequence up to a
  whole block, engines preallocate, and some store the cache in a different dtype
  than the weights. This problem counts tensor data only; treat it as the floor.
- **Shrinking it.** The levers are exactly the factors of the formula: fewer KV heads
  (GQA, and multi-query attention at the limit of one), fewer tokens per layer
  (sliding windows; Gemma 3), smaller elements (8-bit cache quantization in
  `llama.cpp` and `mlx-lm`, which also stores scales that the `"int8"` row here
  ignores), and fewer cached layers (architectures that share one cache across
  layers).
