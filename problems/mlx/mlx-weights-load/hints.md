`mx.load(path)` exposes tensor names, shapes, and dtypes without reading any data (`value.shape` and `value.dtype` do not trigger evaluation). Which of the seven rules can you check using only names and shapes before touching `module` or reading a single byte?

---

Flatten the module's parameters once with `current = dict(tree_flatten(module.parameters()))`. Loop over the file's `(key, value)` pairs, strip `prefix` by checking `key.startswith(prefix)` and slicing (avoid `str.lstrip`, which removes a set of characters rather than a prefix), and look up the remainder in `current`. Collect matched tensors in a `loaded` dict and unmatched original keys in `unexpected`. Calling `value.T` and `value.astype(current[name].dtype)` only adds lazy graph nodes on top of the pending `Load`. Find the `nn.Linear` weights by checking `isinstance(child, nn.Linear)` across `module.named_modules()` and recording `f"{name}.weight"` (or `"weight"` if `name` is empty).

---

Validate everything first, then update the module, then evaluate:

```python
missing = sorted(set(current) - set(loaded))
if strict and (missing or unexpected):
    raise ValueError(...)
module.update(tree_unflatten(list(loaded.items())))
mx.eval(module.parameters())
```

`tree_unflatten` turns `[("layers.0.q_proj.weight", w), ...]` back into the nested dicts and lists that `Module.update` expects, and a partial tree updates only the leaves it contains. Evaluating `module.parameters()` instead of the dict from `mx.load` ensures unexpected tensors stay on disk.
