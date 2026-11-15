There are three moments at which bytes could be read: when the checkpoint is opened,
when a tensor is first asked for, and when it is asked for again. Which bytes does
each moment need, and what do you have to remember so that the third moment needs
none?

---

In the constructor, slice `file[0:8]` for the length $n$ and `file[8 : 8 + n]` for
the JSON, and keep the parsed table and `data_start = 8 + n`. Keep a dict of arrays
already built. `get` looks there first; only on a miss does it slice
`file[data_start + begin : data_start + end]`, which is the one read the tensor
costs. `np.frombuffer` on that slice gives a view without copying. `load_subset`
should check every name against the table before it calls `get` for any of them.

---

Two cases need more than the view. For `BF16`, read the bytes as `"<u2"` and decode:

```python
flat = (bits.astype(np.uint32) << 16).view(np.float32)
```

For any other dtype, the view is only usable if the tensor's offset in the file,
`data_start + begin`, is a multiple of `dtype.itemsize`. If it is not, `flat.copy()`
moves the elements into memory numpy allocated, which is aligned. Reshape last, and
store the result in the cache before returning it.
