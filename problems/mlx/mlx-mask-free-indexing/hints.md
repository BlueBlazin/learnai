In both functions the *shape* of what you need is known before any number is: "the
indices of the `V - k` smallest entries of each row" is always a `(B, V - k)` array,
and the ids to penalize are given. Which MLX functions take an array of indices along
an axis and read or write at those positions? And which function returns indices such
that the `k` largest come first, without sorting everything?

---

`mx.argpartition(a, kth, axis)` returns indices such that position `kth` holds the
element a full sort would put there, with everything smaller before it and everything
larger after it, in no particular order. Partition the *negated* logits around
position `k - 1`: the first `k` indices are then the `k` largest logits, and the
remaining `V - k` are the ones to mask. `mx.put_along_axis(a, indices, values, axis)`
returns a copy of `a` with `values` written at `indices`; it does not touch `a`.

For the penalty, read the current logits at `ids` with `mx.take_along_axis`, subtract,
and write the results back with `mx.put_along_axis`.

---

```python
order = mx.argpartition(-logits, kth=k - 1, axis=-1)
out = mx.put_along_axis(logits, order[..., k:], mx.array(-math.inf), axis=-1)
```

```python
picked = mx.take_along_axis(logits, ids, axis=-1)
out = mx.put_along_axis(logits, ids, picked - value, axis=-1)
```

Why duplicates are harmless in the second: two occurrences of an id read the same
logit, compute the same difference, and write the same number to the same place. It
does not matter which write wins. Handle $k \ge V$ before partitioning: `kth` must be
a valid position.
