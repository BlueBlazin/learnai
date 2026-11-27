Think about what state flows upward from one layer to the next. A dense stack passes only the hidden state `h`. This stack passes a second value as well. Identify that value, which layers overwrite it, and which layers only read it.

---

Keep one variable `idx` for the active selection and loop over the layers in order. A `"full"` layer overwrites `idx` with `dsa.indexer(i, h, k)` using the `h` entering that layer. A `"shared"` layer leaves `idx` unchanged. Every layer then computes `h = h + dsa.attend(i, h, idx)`.

Finding the nearest full layer before layer `i` needs no search, because `idx` already holds the selection written by the most recent full layer.

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

Seeing `idx is None` at a `"shared"` layer means no `"full"` layer has run yet, so raise `ValueError` right there before calling `dsa.attend`. Write `h = h + ...` rather than `h += ...` so you do not mutate the caller's input tensor in PyTorch.
