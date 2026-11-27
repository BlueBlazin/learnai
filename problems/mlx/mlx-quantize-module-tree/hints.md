Think about two details before writing code. First, where can a child module live inside an `nn.Module` besides directly in an attribute? Second, right after `mx.quantize` returns, has any computation happened yet, and what arrays does the result still reference?

---

MLX already handles walking module trees that contain nested lists and dicts. `model.leaf_modules()` returns all modules without children in a nested structure of dicts and lists that mirrors the model, and every `nn.Linear` and `nn.Embedding` is a leaf. `tree_map_with_path(fn, tree, is_leaf=nn.Module.is_module)` calls `fn(path, module)` with each leaf's dotted path and returns a matching tree of results, which `model.update_modules(new_tree)` writes back into the model. Have `fn` return either the unchanged module or `module.to_quantized(group_size=..., bits=...)`, and finish with `mx.eval(model.parameters())`.

---

For `bits_per_weight`, start by summing `nbytes` and `size` across `tree_flatten(model.parameters())`. That weight count is already accurate for every unquantized parameter, so adjust it only for quantized layers:

```python
for _, module in model.named_modules():
    if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        stored = module.weight.size + module.scales.size + module.biases.size
        weights += module.weight.size * 32 // module.bits - stored
```

Each packed `uint32` word holds `32 / bits` codes, so `weight.size * 32 // bits` equals `out * in`.
