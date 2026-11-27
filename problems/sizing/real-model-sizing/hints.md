Three of the four metrics read different views of the same header. `tensor_bytes` uses only `data_offsets`, while `parameters` and `quantized_bytes` use *logical* shapes. Which entries in a quantized file change shape, which ones are excluded as side data, and how can you identify them using only tensor names, dtypes, and `config.get("quantization")`?

---

Write two helper predicates on a tensor name, using `base, _, leaf = name.rpartition(".")` to split a name like `X.weight` into its prefix `base` and suffix `leaf`:

- A tensor is **packed** if `config` has `"quantization"`, the leaf is `"weight"`, the dtype is `"U32"`, and `f"{base}.scales"` is in `header`.
- A tensor is **side data** if the leaf is `"scales"` or `"biases"`, and `f"{base}.weight"` is in `header` and packed.

Then iterate over `header` once. Skip side data, and skip names starting with `"lm_head."` when `tie_word_embeddings` is true and both `model.embed_tokens.weight` and `lm_head.weight` are present. For a packed weight, multiply its last dimension by `32 // bits`. Add the product of the logical shape to `parameters`, and add its quantized size to `quantized_bytes` based solely on whether the logical shape is 2D.

---

```python
count = math.prod(shape)  # logical shape
parameters += count
if len(shape) == 2:
    quantized_bytes += count + 2 * 2 * shape[0] * (shape[1] // 64)  # codes + scales + biases
else:
    quantized_bytes += 2 * count  # bfloat16

tensor_bytes = sum(info["data_offsets"][1] - info["data_offsets"][0] for info in header.values())
head_dim = config["hidden_size"] // config["num_attention_heads"]
kv = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * 2
```

Select tensors to quantize by rank (`len(shape) == 2`), not by tensor name. For example, Granite 4.0 names its MLP matrices `shared_mlp.input_linear.weight` and `shared_mlp.output_linear.weight`.
