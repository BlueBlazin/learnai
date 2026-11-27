In both functions, the *shape* of every intermediate array is known before any values are computed. The indices of the `V - k` smallest entries in each row always form a `(B, V - k)` array, and the `ids` array is already given. Which MLX operations read or write along an axis using an index array? And which operation puts the `k` largest elements first without doing a full sort?

---

`mx.argpartition(a, kth, axis)` returns indices such that position `kth` holds the element a full sort would place there, with smaller elements before it and larger elements after it in arbitrary order. Partition the *negated* logits around `k - 1` so the first `k` indices correspond to the `k` largest logits and the remaining `V - k` are the ones to mask. `mx.put_along_axis(a, indices, values, axis)` returns a new array with `values` written at `indices` without mutating `a`.

For `scatter_penalty`, gather the current logits at `ids` with `mx.take_along_axis`, subtract `value`, and write the results back with `mx.put_along_axis`.

---

```python
order = mx.argpartition(-logits, kth=k - 1, axis=-1)
out = mx.put_along_axis(logits, order[..., k:], mx.array(-math.inf), axis=-1)
```

```python
picked = mx.take_along_axis(logits, ids, axis=-1)
out = mx.put_along_axis(logits, ids, picked - value, axis=-1)
```

Duplicates are harmless in `scatter_penalty` because repeated occurrences of an id read the same logit, compute the same difference, and write the same value to the same slot, so it does not matter which write wins. Handle $k \ge V$ before calling `mx.argpartition`, since `kth` must be a valid index.
