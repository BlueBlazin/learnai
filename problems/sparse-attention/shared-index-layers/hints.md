Which values does a layer need from below? A dense stack passes one thing up, the
hidden state. This stack passes two. Name the second one, and decide which layers
write it and which only read it.

---

Keep one variable for the current selection and walk the layers in order. A full
layer overwrites it with `dsa.indexer(i, h, k)`, computed from the `h` that enters
that layer. A shared layer leaves it alone. Every layer then does the same thing:
`h = h + dsa.attend(i, h, idx)`.

"Nearest full layer before it" needs no search: at layer `i` the variable already
holds what the most recent full layer wrote.

---

```python
idx = None
for layer, kind in enumerate(types):
    if kind == "full":
        idx = dsa.indexer(layer, h, k)
    elif idx is None:
        raise ValueError(...)
    h = h + dsa.attend(layer, h, idx)
return h
```

`idx is None` at a shared layer means no full layer has run yet, which is the error
case. Write `h = h + ...`, not `h += ...`: the in-place form overwrites the caller's
tensor in PyTorch.
