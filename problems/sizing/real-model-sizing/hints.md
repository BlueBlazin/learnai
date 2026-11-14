Three of the four numbers need different views of the same header. `tensor_bytes`
looks only at `data_offsets`. `parameters` looks only at shapes, but at *logical*
shapes. Which entries of a quantized file change shape, which disappear, and how can
you tell them apart using nothing but names, dtypes, and the `quantization` entry?

---

Write two small predicates on a tensor name, using `name.rpartition(".")` to split
`X.weight` into `X` and `weight`:

- packed: the config has `quantization`, the leaf is `weight`, the dtype is `U32`,
  and `X.scales` is in the header;
- side data: the leaf is `scales` or `biases`, and `X.weight` is in the header and
  packed.

Then one loop over the header. Skip side data. Skip names starting with `lm_head.`
when the model is tied and the header has both the embedding and `lm_head.weight`.
For a packed weight, multiply the last dimension by `32 // bits`. Add the product of
the shape to `parameters`, and add the tensor's quantized size to `quantized_bytes`,
which depends on one thing only: whether the logical shape has two dimensions.

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

Select the tensors to quantize by their rank, not by their names: Granite 4.0 calls
its MLP matrices `shared_mlp.input_linear` and `shared_mlp.output_linear`.
