Track the shape of every intermediate tensor. After the projections, the queries and
keys have shape `(B, T, d_k)`. The scores must have shape `(B, T, T)`, with the query
position on the second axis and the key position on the third. Which two axes of the
keys do you have to swap to get that from a matrix product, and which axis must never
be mixed with any other?

---

The steps, with shapes:

1. `q = x @ w_q.T`, and the same for `k` and `v`: the weights are `(out, in)`, so the
   transpose is needed to map `(B, T, d)` to `(B, T, out)`.
2. `scores = q @ k^T / sqrt(d_k)`, where `k^T` swaps the **last two** axes of `k`
   (never the batch axis). `d_k` is the last dimension of `q`, not of `x`.
3. Hide the future, then take the softmax over the last axis.
4. `y = (attn @ v) @ w_o.T`.

For step 3, the entries to hide are those with key index `j` greater than query index
`i`: the strict upper triangle of the `(T, T)` score matrix. The diagonal stays visible.

---

Hide the future in the **scores**, before the softmax, by setting those entries to
`-inf`:

$$
s_{ij} \leftarrow -\infty \quad \text{for } j > i .
$$

Then $e^{-\infty} = 0$: the hidden positions get weight exactly 0, and the softmax
normalizes over the visible ones only. Zeroing the weights after the softmax is not
the same thing, because the visible weights would no longer sum to 1 and would still
depend on the hidden scores through the denominator. Build the boolean mask once from
an upper-triangular matrix (`triu` with `k=1` / `diagonal=1`) of shape `(T, T)`; it
broadcasts over the batch axis.
