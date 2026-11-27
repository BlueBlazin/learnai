Track the shape of every intermediate tensor. After the projections, the queries and
keys have shape `(B, T, d_k)`. The score matrix must have shape `(B, T, T)`, with the
query position on the second axis and the key position on the third. Which two axes of
the keys do you need to swap for that matrix product, and which axis must never be
mixed with any other?

---

Follow these four steps and verify the shape at each stage:

1. Compute `q = x @ w_q.T`, and do the same for `k` and `v`. The weights have shape
   `(out, in)`, so transposing them maps `(B, T, d)` to `(B, T, out)`.
2. Compute `scores = q @ k^T / sqrt(d_k)`, where `k^T` swaps the **last two** axes of
   `k` and leaves the batch axis untouched. Here `d_k` is the last dimension of `q`, not
   of `x`.
3. Hide future positions, then apply softmax over the last axis.
4. Compute `y = (attn @ v) @ w_o.T`.

In step 3, hide every entry where key index `j` is greater than query index `i`. That
is the strict upper triangle of the `(T, T)` score matrix, leaving the diagonal visible.

---

Mask out future positions in the **scores** before softmax by setting them to `-inf`:

$$
s_{ij} \leftarrow -\infty \quad \text{for } j > i .
$$

Because $e^{-\infty} = 0$, hidden positions get weight 0 and softmax normalizes only
over visible positions. Zeroing weights after softmax is not equivalent, because the
visible weights would no longer sum to 1 and would still depend on hidden scores
through the softmax denominator. Build the boolean mask once using an upper-triangular
matrix (`triu` with `k=1` / `diagonal=1`) of shape `(T, T)`. It broadcasts
automatically across the batch axis.
