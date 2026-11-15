The score of every query against every corpus vector is a `(Q, N)` table. Which
single tensor operation produces the whole table, and what has to be done to the
two inputs first so that the same operation gives cosine similarity?

---

Normalize the rows of both matrices for `"cosine"` (divide by the L2 norm with the
reduced axis kept), then `scores = queries @ corpus.T`. Cap `k` at `N`. For the
selection, look at what your framework offers and read what it says about equal
elements: a top-k routine that returns indices may return *any* of several tied
entries, while a stable sort of the negated scores always puts the lower index
first.

---

A full stable sort is correct: `argsort(-scores, stable)[:, :k]`, then gather the
scores. To avoid sorting the $N - k$ entries you discard, take only the $k$-th best
*value* of each row from the top-k routine (values are unambiguous even when
indices are not) and select by value:

```python
kth = topk(scores, k).values[:, -1:]
above, tied = scores > kth, scores == kth
room = k - above.sum(1, keepdim=True)  # places left for tied entries
keep = above | (tied & (tied.cumsum(1) <= room))  # the lowest-index ties get them
```

`keep` has exactly `k` true entries per row. Their column indices, in ascending
order, followed by a stable descending sort of those `k` scores, give the answer.
JAX's `jax.lax.top_k` is documented as stable and needs none of this.
