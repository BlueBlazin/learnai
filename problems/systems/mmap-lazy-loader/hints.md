Consider the three moments when bytes might be read: when opening the checkpoint, when requesting a tensor for the first time, and when requesting the same tensor again. Which bytes does each moment require, and what should you store so the third moment reads nothing?

---

In `__init__`, slice `file[0:8]` to read the header length $n$ and `file[8 : 8 + n]` to parse the JSON metadata, then store the parsed entries and `data_start = 8 + n`. Keep a dict caching materialized arrays. Have `get` check the cache first. On a cache miss, slice `file[data_start + begin : data_start + end]` and wrap it with `np.frombuffer` to get a zero-copy view. In `load_subset`, validate every name against the header table before calling `get` on any of them.

---

Two cases require more than a direct view. For `BF16`, read the raw bytes into an array `bits` with dtype `"<u2"` and decode them to `float32`:

```python
flat = (bits.astype(np.uint32) << 16).view(np.float32)
```

For any other dtype, a view is aligned only when the tensor's file offset `data_start + begin` is a multiple of `dtype.itemsize`. If it is not, call `.copy()` on the view to move the data into aligned NumPy memory. Reshape at the end, cache the array, and return it.
