The score is an attention score with three changes: the key has no head axis, each
head's dot product goes through a ReLU, and the heads are combined by a weighted sum
with weights that depend on the query. Write the shape of every intermediate tensor
before you write the code: which tensor has shape `(T, H, S)`, and where in the
formula does the ReLU see it?

---

`index_scores`: compute all dot products at once as a `(T, H, S)` tensor
(`einsum("thd,sd->ths", q_idx, k_idx)`), multiply by `D ** -0.5`, apply the ReLU, and
contract the head axis with `w * H ** -0.5` (`einsum("th,ths->ts", ...)`).

`topk_mask`: first make the invisible entries unable to win, by replacing their
scores with `-inf` (not with 0: visible scores can be negative). Then you need the
`k` best of each row with a defined order among equal scores. A top-k routine does not
promise one; a stable sort does.

---

Sort each row's masked scores in descending order with a stable sort, so that equal
scores keep their index order: `torch.argsort(masked, descending=True, stable=True)`,
or `argsort(-masked)` in JAX and MLX, whose sorts are stable. Turn the order into
ranks with a second `argsort` (the rank of entry `s` is the place where `s` appears in
the order). Then

```python
mask = (rank < k) & visible
```

`rank < k` needs no special case for `k > S`. The `& visible` matters for rows with
fewer than `k` visible entries: their ranks below `k` run on into the invisible
entries, which all sit at `-inf`.
