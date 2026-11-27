The index score is like an attention score with three differences. The key has no head axis, each head's dot product passes through a ReLU, and the heads combine via a query-dependent weighted sum. Write down the shape of each intermediate tensor before coding. Which tensor has shape `(T, H, S)`, and where does the ReLU act on it?

---

For `index_scores`, compute all per-head dot products at once as a `(T, H, S)` tensor using `einsum("thd,sd->ths", q_idx, k_idx)`. Multiply by `D ** -0.5`, apply ReLU, and contract the head axis with `w * H ** -0.5` via `einsum("th,ths->ts", ...)`.

For `topk_mask`, first prevent invisible entries from winning by replacing their scores with `-inf`. Do not use `0`, because visible scores can be negative. Next, select the `k` best entries per row with a deterministic tie order. A standard top-k routine does not guarantee tie order, whereas a stable sort does.

---

Sort each row's masked scores in descending order with a stable sort so equal scores preserve their original index order. In PyTorch, use `torch.argsort(masked, descending=True, stable=True)`. In JAX and MLX, sorts are stable by default, so use `argsort(-masked)`. Invert the ordering with a second `argsort` to get each entry's 0-based rank, then intersect with `visible`:

```python
mask = (rank < k) & visible
```

Comparing `rank < k` needs no special handling when `k > S`. Intersecting with `visible` removes invisible `-inf` entries on rows that have fewer than `k` visible keys.
