Size a real checkpoint from two small pieces of text: the header of its safetensors
file and its `config.json`. Four numbers come out: the parameter count, the bytes on
disk, the bytes one token adds to the KV cache, and the size the model would have
after 8-bit quantization. No weight is read. The catch is that a file stores
*tensors*, and tensors are not always parameters: a quantized file packs four weights
into one stored element and adds side tables, and a tied matrix may be stored twice.

## Specification

```python
def size_report(header: dict, config: dict) -> dict[str, int]: ...
```

- `header`: the tensor table of a safetensors file, as `read_header` of
  `safetensors-parse` returns it: a dict from tensor name to
  `{"dtype": str, "shape": list[int], "data_offsets": [begin, end]}`.
- `config`: the parsed `config.json`. The function uses `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `tie_word_embeddings`,
  and `quantization` if present; it ignores every other key.

The result is a dict with exactly the four keys below, all Python ints.

**Logical tensors.** The parameters of the model are the elements of its *logical*
tensors, which are the header's entries after two corrections.

1. *Packed weights.* If `config` has a `"quantization"` entry
   `{"bits": b, "group_size": g, ...}`, the file was written by a quantizer that
   stores a matrix `X.weight` of logical shape `[out, in]` as three tensors:

   | tensor | dtype | stored shape | content |
   |---|---|---|---|
   | `X.weight` | `U32` | `[out, in * b / 32]` | the $b$-bit codes, $32 / b$ per word |
   | `X.scales` | `BF16` | `[out, in / g]` | one scale per group of $g$ weights |
   | `X.biases` | `BF16` | `[out, in / g]` | one bias per group |

   An entry `X.weight` is **packed** when its dtype is `U32` and `X.scales` is also in
   the header. Its logical shape is `[out, words * 32 // b]`. The `X.scales` and
   `X.biases` of a packed weight are **side data**: they are stored, but they are not
   logical tensors. Without a `"quantization"` entry nothing is packed and nothing is
   side data, whatever the names and dtypes.
2. *A tied head stored twice.* If `config["tie_word_embeddings"]` is true and the
   header contains both `model.embed_tokens.weight` and `lm_head.weight`, the head is
   a second copy of the embedding: no entry whose name starts with `lm_head.` is a
   logical tensor.

Every other entry is a logical tensor with the shape in the header.

**`"parameters"`**: the sum over logical tensors of the product of the logical shape
(1 for a scalar, shape `[]`).

**`"tensor_bytes"`**: the bytes of the file's data section, $\sum (\text{end} -
\text{begin})$ over **all** entries of the header, side data and second copies
included. (The file itself is 8 bytes plus the header longer.)

**`"kv_bytes_per_token"`**: the bytes one token of one sequence adds to a bfloat16 KV
cache: a key and a value vector per layer and per key/value head,

$$
2 \cdot \texttt{num\_hidden\_layers} \cdot \texttt{num\_key\_value\_heads} \cdot
\frac{\texttt{hidden\_size}}{\texttt{num\_attention\_heads}} \cdot 2 .
$$

**`"quantized_bytes"`**: the predicted data-section size of the model after 8-bit
quantization with groups of 64, in the layout of the table above, whatever format the
input file is in. Summed over logical tensors:

- a logical tensor with two dimensions `[out, in]` is quantized: `out * in` bytes of
  codes, plus one bfloat16 scale and one bfloat16 bias for each of the
  `out * in / 64` groups, $4 \cdot \text{out} \cdot \text{in} / 64$ bytes;
- a logical tensor with any other number of dimensions (a norm gain) is kept in
  bfloat16: 2 bytes per element.

The function does not modify its arguments.

## Example

A one-block model: vocabulary 100, width 64, two query heads sharing one key/value
head of width 32, a SwiGLU MLP of width 128, tied embeddings, stored in bfloat16.

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
| `tensor_bytes` | every tensor is BF16: $2 \cdot 43456$ | 86,912 |
| `kv_bytes_per_token` | $2 \cdot 1 \cdot 1 \cdot 32 \cdot 2$ | 128 |
| `quantized_bytes` | $43264 + 4 \cdot 43264 / 64 + 2 \cdot 192$ | 46,352 |

Now suppose the same file also stored `lm_head.weight`, shape `[100, 64]`. The model
is tied, so the head is a copy: `parameters` and `quantized_bytes` do not change, and
`tensor_bytes` grows by 12,800 to 99,712. With `tie_word_embeddings: False` the head
is a matrix of its own: 49,856 parameters and 53,152 quantized bytes.

The same model as an 8-bit file has, for each matrix, a `U32` weight with a quarter
of the columns plus scales and biases: `q_proj.weight` is `[64, 16]` and
`q_proj.scales` is `[64, 1]`. Its `parameters` are still 43,456, and its
`tensor_bytes` are now 46,352: the prediction, come true.

## Constraints

- Results are compared exactly.
- Dtypes are among `BF16`, `F16`, `F32`, `U32`, `I64`. `bits` is 2, 4, or 8. Every
  logical matrix has a second dimension that is a multiple of 64, and
  `num_attention_heads` divides `hidden_size`.
- Real inputs: the header and config of `ibm-granite/granite-4.0-350m` (226 tensors,
  bfloat16, tied, read from the downloaded checkpoint) and of the 8-bit MLX build of
  Granite 4.2 3B, `lmstudio-community/granite-4.2-3b-MLX-8bit` (927 tensors, untied;
  its header is shipped with the problem, so this case needs no download). The
  expected numbers were measured with `transformers` and `mlx-lm`: 352,379,904 and
  3,659,737,600 parameters, and for the small model the exact size of the file that
  `mlx-lm` writes when asked to quantize it to 8 bits.

## Why this matters

Before a model is loaded, these four numbers decide whether it can be: the bytes on
disk are the resident memory of the weights, the KV bytes per token times context
times concurrency is the memory that grows with traffic, and the quantized size says
what a smaller build would buy. All of them can be read off a header fetched with one
HTTP range request, without downloading gigabytes. The traps are the ones in the
specification. Counting the stored elements of an 8-bit file reports 1.03 billion
parameters for a 3.66 billion parameter model. A file that stores a tied matrix under
both names advertises a quarter more parameters than the model has. And "8-bit" is
8.5 bits per weight once the scales and biases are counted, which is 229 MB on a
3.9 GB file.
