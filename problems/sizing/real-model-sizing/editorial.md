## The idea

In `param-count`, you derived a model's size from its architecture hyperparameters. Here you work in the opposite direction and read the size directly from the checkpoint artifact. That is the practical approach when inspecting a model you did not design, because the safetensors header lists every tensor that actually exists. You do not have to guess about fused MLP matrices, omitted biases, or extra normalization layers. However, the header only records stored tensors, so you must undo two storage conventions to recover the model's logical tensors.

**Packing.** MLX affine quantization (like several other formats) packs $b$-bit integer codes along each row into 32-bit unsigned integers. An `[out, in]` matrix at 8 bits is therefore stored as a `U32` tensor of shape `[out, in / 4]`. Alongside it are a bfloat16 scale and a bfloat16 bias for each group of $g$ consecutive weights, reconstructing each weight as $w = \text{code} \cdot \text{scale} + \text{bias}$. The packed codes represent the model's parameters, while the scales and biases are storage overhead. At 8 bits with groups of 64, that overhead is $2 \cdot 16 / 64 = 0.5$ bits per weight:

$$
\text{bytes}(W) = \underbrace{\text{out} \cdot \text{in}}_{\text{codes}}
+ \underbrace{2 \cdot 2 \cdot \text{out} \cdot \tfrac{\text{in}}{64}}_{\text{scales and biases}}
= \tfrac{17}{16}\, \text{out} \cdot \text{in}.
$$

**Tying.** A model with tied embeddings shares a single matrix of shape `[vocab_size, hidden_size]` between the input embedding and the output head. A safetensors file written from a tied model usually omits `lm_head.weight` completely, as Granite 4.0 does. Some exporters still write both tensors with identical values, and only `tie_word_embeddings` in `config.json` distinguishes a duplicate copy from an independent output matrix.

With those two rules in place, a single loop handles both bfloat16 checkpoints and already-quantized checkpoints. This also gives a built-in consistency check. For the 8-bit checkpoint of Granite 4.2 3B, `quantized_bytes` must match `tensor_bytes` to the exact byte (3,888,665,600), because the file on disk already uses the target 8-bit group-64 layout.

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

The 350M model's `quantized_bytes` prediction matches `mlx_lm.convert` exactly. The converted file's data section is 374,458,368 bytes, and the full file is the 374.5 MB published 8-bit build. The KV cache numbers were verified by measuring the bytes held by each library's cache after a forward pass and dividing by the number of tokens fed. At 28 kB per token, the 350M model's full 32,768-token context takes 940 MB of KV cache, which is larger than its weights.

## Common mistakes

- **Counting a tied head twice.** Summing every tensor in a file that stores both `model.embed_tokens.weight` and `lm_head.weight` double-counts the embedding. Qwen3-0.6B's checkpoint stores both, which causes the Hugging Face Hub counter to report 751.6 M parameters for a 596.0 M model.
- **Dropping an untied head.** Always skipping `lm_head.weight` just because it has the same shape as the embedding matrix drops real parameters. Granite 4.2 3B is untied, so its output head contributes 257 M parameters of its own. Always check `tie_word_embeddings` in `config`.
- **Counting stored elements directly.** Running `sum(prod(shape))` over an 8-bit file returns 1.03 B for the 3B model because each packed `U32` word holds four weights, while the scales and biases get counted as if they were parameters.
- **Forgetting quantization overhead (or biases).** Estimating 1 byte per weight predicts 3.66 GB for a 3.89 GB file. Counting only scales and forgetting biases makes the same error at half the magnitude.
- **Quantizing norm gains.** Quantizers leave 1D tensors in bfloat16. On the 350M model that difference is only 58 kB, which is easy to miss without an exact check.
- **Selecting matrices by name.** Filtering with `name.endswith("_proj.weight")` works for Llama, Qwen, and Granite 4.2, but fails on Granite 4.0. Granite 4.0 fuses its SwiGLU gate and up projections into `shared_mlp.input_linear.weight` and names the down projection `shared_mlp.output_linear.weight`. A name filter leaves 176 M parameters at 16 bits and predicts 540 MB instead of 374 MB. Tensor rank (`len(shape) == 2`) matches how the quantizer selects linear and embedding layers.
- **Computing bytes from shapes.** Multiplying `2 * parameters` only works for pure 16-bit files without duplicates. It fails on float32 tensors, packed quantized files, and duplicated heads. Always sum `data_offsets` for `tensor_bytes`.
- **Using query heads or omitting values in the KV cache.** The cache stores both keys and values for each key/value head, not each query head. Granite 4.0 350M has 4 KV heads, not 16.

## Complexity

The algorithm makes one pass over the header in $O(n)$ time for $n$ tensors, typically a few hundred to a few thousand entries.

## In the wild

The Hugging Face Hub model page computes its parameter count from the safetensors header alone, fetched with an HTTP range request for the first few hundred kilobytes of the file. That is why it can display per-dtype counts without downloading weights, and why naive counting misreports quantized or duplicated checkpoints. After converting the 350M model, `mlx_lm` reports "8.501 bits per weight", which is `8 * quantized_bytes / parameters` including the unquantized norm gains. GGUF files store equivalent metadata in a different header format, and `llama.cpp` prints a similar sizing report at load time. For capacity planning, add 8 bytes plus the header length to `tensor_bytes` for the file size, and multiply `kv_bytes_per_token` by context length and batch size to estimate KV cache memory.
