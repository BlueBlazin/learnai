## The idea

`param-count` derived the size of a model from its architecture. This problem goes
the other way and reads it off the artifact, which is what you do with a model you
did not design: the header lists every tensor that exists, so there is nothing to
guess about fused MLP matrices, missing biases, or extra norms. What the header does
not tell you is what a stored tensor *means*. Two conventions have to be undone.

**Packing.** MLX's affine quantization (and it is not alone) stores the $b$-bit codes
of a row packed into 32-bit words, so an `[out, in]` matrix at 8 bits appears as a
`U32` tensor of shape `[out, in / 4]`. Next to it are a scale and a bias per group of
$g$ consecutive weights, with $w = \text{code} \cdot \text{scale} + \text{bias}$.
The codes are the parameters; the scales and biases are the price of representing
them. For 8 bits and groups of 64 the price is $2 \cdot 16 / 64 = 0.5$ bits per
weight:

$$
\text{bytes}(W) = \underbrace{\text{out} \cdot \text{in}}_{\text{codes}}
+ \underbrace{2 \cdot 2 \cdot \text{out} \cdot \tfrac{\text{in}}{64}}_{\text{scales and biases}}
= \tfrac{17}{16}\, \text{out} \cdot \text{in}.
$$

**Tying.** A tied model has one $V \times d$ matrix. safetensors stores each tensor
once, and a file written from a tied model normally has no `lm_head.weight` at all,
as Granite 4.0's does not. But a file *can* contain both names with the same values,
and then only the config can tell a copy from a second matrix.

With those two rules the loop is a few lines, and it applies unchanged to a bfloat16
file and to a file that is already quantized. That gives a free check: for the 8-bit
checkpoint of Granite 4.2 3B, `quantized_bytes` must equal `tensor_bytes`, because the
file on disk is exactly the quantization being predicted. It does, to the byte:
3,888,665,600.

## Reference solution (Python)

```python
def size_report(header, config):
    quant = config.get("quantization")

    def is_packed(name):
        base, _, leaf = name.rpartition(".")
        return (
            quant is not None
            and leaf == "weight"
            and header[name]["dtype"] == "U32"
            and f"{base}.scales" in header
        )

    def is_side_data(name):
        base, _, leaf = name.rpartition(".")
        weight = f"{base}.weight"
        return leaf in ("scales", "biases") and weight in header and is_packed(weight)

    duplicate_head = (
        bool(config["tie_word_embeddings"])
        and "model.embed_tokens.weight" in header
        and "lm_head.weight" in header
    )

    parameters = quantized_bytes = 0
    for name, info in header.items():
        if is_side_data(name) or (duplicate_head and name.startswith("lm_head.")):
            continue
        shape = list(info["shape"])
        if is_packed(name):
            shape[-1] = shape[-1] * 32 // quant["bits"]
        count = prod(shape)
        parameters += count
        if len(shape) == 2:
            quantized_bytes += count + 2 * 2 * shape[0] * (shape[1] // 64)
        else:
            quantized_bytes += 2 * count

    tensor_bytes = sum(i["data_offsets"][1] - i["data_offsets"][0] for i in header.values())
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    kv = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * 2
    return {
        "parameters": parameters,
        "tensor_bytes": tensor_bytes,
        "kv_bytes_per_token": kv,
        "quantized_bytes": quantized_bytes,
    }
```

## The two Granite checkpoints

| | Granite 4.0 350M (BF16) | Granite 4.2 3B (8-bit MLX) |
|---|---|---|
| tensors in the file | 226 | 927 |
| stored elements | 352,379,904 | 1,029,450,240 |
| `parameters` | 352,379,904 | 3,659,737,600 |
| `tensor_bytes` | 704,759,808 | 3,888,665,600 |
| `kv_bytes_per_token` | 28,672 | 81,920 |
| `quantized_bytes` | 374,458,368 | 3,888,665,600 |

The small model's 8-bit prediction is what `mlx_lm.convert` produces: the data section
of the file it writes is 374,458,368 bytes, and the whole file is the 374.5 MB that
the published 8-bit build weighs. The KV figures were measured too, as the bytes held
by each library's own cache after a forward pass divided by the tokens fed. At 28 kB
per token the small model's full 32,768-token context costs 940 MB of cache, more
than its weights.

## Common mistakes

- **Counting a tied head twice.** Summing every tensor in a file that stores both
  names. Qwen3-0.6B's checkpoint does this, and the Hub's counter reports 751.6 M for
  a 596.0 M model.
- **Dropping an untied head.** The opposite: treating `lm_head.weight` as a copy
  because it has the embedding's shape. Granite 4.2 3B is untied, and its head is
  257 M parameters of its own. The config decides.
- **Counting stored elements.** `sum(prod(shape))` over an 8-bit file gives 1.03 B
  for the 3B model: the packed words count a quarter of the weights, and the scales
  and biases are added as if they were parameters.
- **Forgetting the quantization overhead**, or half of it. One byte per weight
  predicts 3.66 GB for a file of 3.89 GB. Counting scales without biases is the same
  mistake at half size.
- **Quantizing the norm gains.** Quantizers skip one-dimensional tensors. It is
  only 58 kB for the small model, which is why an exact comparison is useful.
- **Selecting matrices by name.** `name.endswith("_proj.weight")` covers Llama, Qwen,
  and Granite 4.2. Granite 4.0 fuses the SwiGLU gate and up projections into
  `shared_mlp.input_linear.weight` and calls the down projection
  `shared_mlp.output_linear.weight`; a name filter leaves 176 M parameters at 16 bits
  and predicts 540 MB instead of 374 MB. Rank is the criterion the quantizer itself
  uses (it walks the modules and converts every linear and embedding layer).
- **Bytes from shapes.** `2 * parameters` is right for a pure BF16 file and wrong
  for float32, for packed files, and for files with duplicates. The offsets are the
  truth about the disk.
- **Query heads in the KV cache**, or keys without values. The cache holds what the
  key and value projections produce, per KV head: 4 heads, not 16, for the small
  model.

## Complexity

One pass over the header: $O(n)$ for $n$ tensors, a few hundred to a few thousand.

## In the wild

The Hub's model page computes its parameter count this way, from the safetensors
header alone, fetched with a range request for the first few hundred kilobytes of the
file; that is why it shows per-dtype counts and why it miscounts quantized and
duplicated tensors. `mlx_lm` reports "8.501 bits per weight" after converting the
small model: that is `8 * quantized_bytes / parameters` with the norm gains included.
GGUF files carry the same information in a different header, and llama.cpp prints
the equivalent report at load time. For capacity planning, add the 8 bytes and the
header to `tensor_bytes` for the file size, and multiply `kv_bytes_per_token` by
context length and batch size for the memory that `kv-cache-bytes` taught you to
watch.
