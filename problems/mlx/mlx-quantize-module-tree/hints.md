Two questions. Where can a child module live inside an `nn.Module`, besides directly
in an attribute? And when `mx.quantize` has returned, has anything been computed, and
what does the result still hold on to?

---

MLX already knows how to enumerate a module tree with its containers.
`model.leaf_modules()` returns the modules that have no children, in a nested
structure of dicts and lists that mirrors the tree; every `nn.Linear` and
`nn.Embedding` is such a leaf. `tree_map_with_path(fn, tree, is_leaf=nn.Module.is_module)`
calls `fn(path, module)` with the dotted path of each leaf and builds the same
structure out of what `fn` returns, and `model.update_modules(new_tree)` puts that
structure back into the model. Your `fn` returns the module unchanged, or
`module.to_quantized(group_size=..., bits=...)`. Finish with
`mx.eval(model.parameters())`.

---

For `bits_per_weight`, add up `nbytes` and `size` over
`tree_flatten(model.parameters())`. That count of weights is right for everything
except quantized layers, so correct it for each of them:

```python
for _, module in model.named_modules():
    if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        stored = module.weight.size + module.scales.size + module.biases.size
        weights += module.weight.size * 32 // module.bits - stored
```

A packed word holds `32 / bits` codes, so `weight.size * 32 // bits` is `out * in`.
