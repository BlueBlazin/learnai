Implement `size_report` to compute four sizing metrics for a model checkpoint from its safetensors header and `config.json`, without loading any weights. Stored tensors do not always map one-to-one to model parameters, because a quantized file packs multiple weights into each stored element alongside scale and bias tables, and a tied output head may be stored as a second copy of the embedding matrix.

## Specification

```python
def size_report(header: dict, config: dict) -> dict[str, int]: ...
```

- `header` is the tensor table of a safetensors file, in the format returned by `read_header` in `safetensors-parse`. It maps each tensor name to `{"dtype": str, "shape": list[int], "data_offsets": [begin, end]}`.
- `config` is the parsed `config.json`. Use `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `tie_word_embeddings`, and `quantization` (when present). Ignore all other keys.

Return a dict with exactly four keys, all mapped to Python `int` values.

### Logical tensors

The model's parameters are the elements of its *logical* tensors, obtained from `header` after two adjustments.

1. **Packed weights.** When `config` contains a `"quantization"` dict `{"bits": b, "group_size": g, ...}`, the checkpoint was written by a quantizer that stores a logical weight matrix `X.weight` of shape `[out, in]` as three tensors.

   | tensor | dtype | stored shape | content |
   |---|---|---|---|
   | `X.weight` | `U32` | `[out, in * b / 32]` | $b$-bit codes, $32 / b$ per word |
   | `X.scales` | `BF16` | `[out, in / g]` | one scale per group of $g$ weights |
   | `X.biases` | `BF16` | `[out, in / g]` | one bias per group |

   An entry `X.weight` is **packed** if its dtype is `U32` and `X.scales` is also present in `header`. If its stored shape is `[out, cols]`, its logical shape is `[out, cols * 32 // b]`. The corresponding `X.scales` and `X.biases` entries are **side data**. They occupy bytes on disk, but they are not logical tensors. When `config` has no `"quantization"` entry, nothing is packed and nothing is side data, regardless of tensor names or dtypes.

2. **Duplicated tied head.** If `config["tie_word_embeddings"]` is true and `header` contains both `model.embed_tokens.weight` and `lm_head.weight`, the output head is a second copy of the embedding matrix. Exclude every entry whose name starts with `lm_head.` from the logical tensors.

Every remaining entry in `header` is a logical tensor with its stored shape.

### Output keys

- **`"parameters"`** is the sum over all logical tensors of the product of each logical shape (with a scalar of shape `[]` having product 1).
- **`"tensor_bytes"`** is the total size of the file's data section, computed as $\sum (\text{end} - \text{begin})$ across **all** entries in `header`, including side data and duplicated tied heads. (The full file on disk also includes an 8-byte length prefix and the JSON header.)
- **`"kv_bytes_per_token"`** is the number of bytes that one token of one sequence adds to a bfloat16 KV cache. Each layer stores one key vector and one value vector per key/value head, where each head has width `hidden_size / num_attention_heads` and each bfloat16 value takes 2 bytes:

  $$
  2 \cdot \texttt{num\_hidden\_layers} \cdot \texttt{num\_key\_value\_heads} \cdot
  \frac{\texttt{hidden\_size}}{\texttt{num\_attention\_heads}} \cdot 2 .
  $$

- **`"quantized_bytes"`** is the predicted data-section size after 8-bit quantization with group size 64 in the layout above, regardless of the input file's current format. Sum the following over all logical tensors:
  - A 2D logical tensor of shape `[out, in]` is quantized. It takes `out * in` bytes for the 8-bit codes, plus one bfloat16 scale (2 bytes) and one bfloat16 bias (2 bytes) for each of the `out * in / 64` groups, adding $4 \cdot \text{out} \cdot \text{in} / 64$ bytes.
  - A logical tensor with any other number of dimensions (such as a 1D normalization gain) stays in bfloat16 at 2 bytes per element.

Do not modify `header` or `config`.

## Example

Consider a one-block bfloat16 model with vocabulary size 100, hidden size 64, 2 query heads sharing 1 key/value head of width 32, a SwiGLU MLP of width 128, and tied embeddings.

```python
header = {
    "model.embed_tokens.weight":                     {"dtype": "BF16", "shape": [100, 64], "data_offsets": [0, 12800]},
    "model.layers.0.input_layernorm.weight":         {"dtype": "BF16", "shape": [64], ...},
    "model.layers.0.self_attn.q_proj.weight":        {"dtype": "BF16", "shape": [64, 64], ...},
    "model.layers.0.self_attn.k_proj.weight":        {"dtype": "BF16", "shape": [32, 64], ...},
    "model.layers.0.self_attn.v_proj.weight":        {"dtype": "BF16", "shape": [32, 64], ...},
    "model.layers.0.self_attn.o_proj.weight":        {"dtype": "BF16", "shape": [64, 64], ...},
    "model.layers.0.post_attention_layernorm.weight": {"dtype": "BF16", "shape": [64], ...},
    "model.layers.0.mlp.gate_proj.weight":           {"dtype": "BF16", "shape": [128, 64], ...},
    "model.layers.0.mlp.up_proj.weight":             {"dtype": "BF16", "shape": [128, 64], ...},
    "model.layers.0.mlp.down_proj.weight":           {"dtype": "BF16", "shape": [64, 128], ...},
    "model.norm.weight":                             {"dtype": "BF16", "shape": [64], ...},
}
config = {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
          "num_key_value_heads": 1, "tie_word_embeddings": True, ...}

size_report(header, config)
# {"parameters": 43456, "tensor_bytes": 86912, "kv_bytes_per_token": 128, "quantized_bytes": 46352}
```

| quantity | arithmetic | value |
|---|---|---|
| matrices | $6400 + 4096 + 2 \cdot 2048 + 4096 + 3 \cdot 8192$ | 43,264 |
| norm gains | $3 \cdot 64$ | 192 |
| `parameters` | $43264 + 192$ | 43,456 |
| `tensor_bytes` | $2 \cdot 43456$ (all tensors are BF16) | 86,912 |
| `kv_bytes_per_token` | $2 \cdot 1 \cdot 1 \cdot 32 \cdot 2$ | 128 |
| `quantized_bytes` | $43264 + 4 \cdot 43264 / 64 + 2 \cdot 192$ | 46,352 |

Now suppose the same file also contains `lm_head.weight` with shape `[100, 64]`. Because `tie_word_embeddings` is true, the head is a duplicate copy. Both `parameters` and `quantized_bytes` stay unchanged, while `tensor_bytes` increases by 12,800 to 99,712. If `tie_word_embeddings` is false, the head is a separate matrix, raising `parameters` to 49,856 and `quantized_bytes` to 53,152.

If the same model is stored as an 8-bit quantized file, each matrix has a `U32` weight with one quarter as many columns, plus `scales` and `biases` tensors. For example, `q_proj.weight` has shape `[64, 16]` and `q_proj.scales` has shape `[64, 1]`. Its `parameters` remain 43,456, and its `tensor_bytes` drop to 46,352, matching the earlier `quantized_bytes` prediction.

## Constraints

- Results are compared exactly.
- Tensor dtypes are among `BF16`, `F16`, `F32`, `U32`, and `I64`.
- `bits` is 2, 4, or 8.
- Every 2D logical tensor has a second dimension divisible by 64.
- `num_attention_heads` divides `hidden_size`.
- Real-model test cases use the header and config of `ibm-granite/granite-4.0-350m` (226 tensors, bfloat16, tied, read from the downloaded checkpoint) and the 8-bit MLX build of Granite 4.2 3B, `lmstudio-community/granite-4.2-3b-MLX-8bit` (927 tensors, untied; its header is shipped with the problem so no download is required). Expected values were measured with `transformers` and `mlx-lm`, giving 352,379,904 and 3,659,737,600 parameters, and matching the exact data-section size that `mlx-lm` writes when quantizing the 350M model to 8 bits.

## Why this matters

Before loading a model, these four numbers determine whether it fits in memory. `tensor_bytes` gives the resident weight footprint, `kv_bytes_per_token` multiplied by context length and concurrency gives the traffic-dependent cache memory, and `quantized_bytes` shows how much an 8-bit build would save. All four can be computed from a header fetched with a single HTTP range request, without downloading gigabytes of weights.

Naive counters fail silently on real checkpoints. Summing stored tensor elements on an 8-bit checkpoint reports 1.03 billion parameters for a 3.66 billion parameter model. Counting a duplicated tied head inflates the parameter count by roughly a quarter. Ignoring scales and biases underestimates an 8-bit checkpoint because 8-bit group-64 quantization actually uses 8.5 bits per weight, adding 229 MB of overhead to a 3.9 GB file.
