Every query-corpus pair produces one entry in a `(Q, N)` score table. Think about which single tensor operation computes that entire table at once, and what preprocessing turns that same operation into cosine similarity.

---

For `"cosine"`, divide each row of `queries` and `corpus` by its L2 norm (keeping the reduced dimension), then compute `scores = queries @ corpus.T`. Cap `k` at `N`.

For selecting the top `k`, check whether your framework's selection routine is stable on ties. An unstable top-k may return any index among tied entries, whereas a stable sort of `-scores` always places lower indices first.

---

A full stable sort gives the exact answer: `argsort(-scores, stable)[:, :k]`, followed by gathering the corresponding scores. To avoid sorting all $N - k$ discarded entries, take only the $k$-th best *value* per row from `topk` (values are unambiguous even when indices are not) and select by value:

```python
kth = topk(scores, k).values[:, -1:]
above, tied = scores > kth, scores == kth
room = k - above.sum(1, keepdim=True)  # places left for tied entries
keep = above | (tied & (tied.cumsum(1) <= room))  # the lowest-index ties get them
```

`keep` has exactly `k` true entries per row. Extracting their column indices in ascending order and running a stable descending sort on those `k` scores completes the search. In JAX, `jax.lax.top_k` is already stable and needs no extra tie-breaking step.
