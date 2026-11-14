`mx.load(path)` gives you names, shapes, and dtypes for free: `value.shape` and
`value.dtype` are known without evaluating anything. Which of the seven rules can be
decided from names and shapes alone, before any tensor is read, and before the module
is touched?

---

Flatten the module once: `current = dict(tree_flatten(module.parameters()))` maps
every parameter path to its array. Walk the file's items: strip the prefix (check
`key.startswith(prefix)` and slice; `str.lstrip` removes a *set of characters*, not a
prefix), look the remainder up in `current`, and collect the tensors that fit into a
dict and the others into `unexpected`. Transposing (`value.T`) and casting
(`value.astype(current[name].dtype)`) only add graph nodes on top of the pending
load. The Linear weights are `f"{name}.weight"` for every `(name, child)` of
`module.named_modules()` with `isinstance(child, nn.Linear)`.

---

Validate everything first, then update, then evaluate, in that order:

```python
missing = sorted(set(current) - set(loaded))
if strict and (missing or unexpected):
    raise ValueError(...)
module.update(tree_unflatten(list(loaded.items())))
mx.eval(module.parameters())
```

`tree_unflatten` turns `[("layers.0.q_proj.weight", w), ...]` back into the nested
dicts and lists that `Module.update` expects; a partial tree updates only the leaves it
has. Evaluating `module.parameters()`, and not the dict that `mx.load` returned, is
what leaves the unexpected tensors on disk.
