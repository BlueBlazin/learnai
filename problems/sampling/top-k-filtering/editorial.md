## The idea

"The k largest" is a statement about ranks, not about values. A value threshold
answers a different question (which entries are at least as large as the k-th one),
and the two differ exactly when there is a tie at the boundary. So compute ranks.

Sort each row in decreasing order with a stable sort. Stability is what implements the
tie rule: among equal logits, the original (index) order is preserved, so the lowest
index gets the best rank. The sort returns `order`, mapping rank to token. Inverting
that permutation gives `rank`, mapping token to rank, and ranks are distinct numbers
$0, \ldots, V - 1$, so `rank < k` is true for exactly `k` entries per row.

Masked entries need no special handling. `-inf` sorts after every finite value, so a
masked token is kept only when `k` exceeds the number of finite entries, and "keeping"
it means copying `-inf`.

## Reference solution (PyTorch)

```python
def top_k_filter(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return torch.where(rank < k, logits, float("-inf"))
```

`argsort(order)` inverts the permutation: `order[b, r] = v` means token `v` has rank
`r`, and sorting the tokens `order[b, :]` into increasing order puts position `r` at
slot `v`. An alternative that avoids the second sort is a scatter, as in
`top-p-filtering`: `keep = zeros(B, V, bool).scatter(-1, order[:, :k], True)`.

In JAX and MLX, `argsort` is stable and has no `descending` argument; `argsort(-logits)`
gives the same ranking, because negation maps equal values to equal values and the
stable sort then keeps them in index order.

The early return carries both conventions. Without the `k <= 0` test, `rank < 0` is
false everywhere and the whole batch is masked.

## Common mistakes

- **A value threshold.** `logits >= kth_largest` keeps every entry tied with the k-th
  one. With all-equal logits and `k = 1` it keeps the whole row. This is the most
  widespread formulation (it is what Hugging Face's `TopKLogitsWarper` does), and it
  is fine as a sampler; it just is not "exactly k", and code downstream that assumes
  at most `k` candidates is then wrong.
- **Reading `k = 0` literally.** An empty candidate set leaves a row of `-inf`, whose
  softmax is NaN. Clients send 0 (or leave the field at its default of 0) to mean "no
  top-k".
- **Not allowing `k > V`.** `topk(logits, k)`, `sorted[:, k - 1]`, and `kthvalue` all
  raise when `k` exceeds the vocabulary size. It happens in production whenever a
  default `k` meets a small vocabulary or an already filtered candidate list.
- **Reversed ties.** Sorting ascending and flipping gives a descending order, but with
  equal entries in reverse index order. Use a descending stable sort, or sort the
  negated logits.
- **Off by one**: `rank <= k`, or `order[:, : k + 1]`.
- **Filling with a large negative number** (`-1e9`, `finfo.min`) instead of `-inf`.
  The token is then still a candidate: a later filter or a "keep at least `n`" rule
  counts it as live, and every check of the form "is this token masked" (`== -inf`)
  says no.
- **One ranking for the whole batch.** Flattening before the sort (or reducing the
  threshold to one scalar) keeps the `k` largest logits overall: rows with large
  logits keep everything and the others nothing.

## Complexity

$O(B \cdot V \log V)$ time for the sort and $O(B \cdot V)$ memory. Only the top `k`
need to be found, not ordered, so engines use a partial selection: `topk` or
`argpartition`, $O(V + k \log k)$, at the price of an unspecified order among ties.

## In the wild

llama.cpp's `llama_sampler_top_k_impl` returns immediately for `k <= 0`, clamps `k`
to the number of candidates, partially sorts the candidates, and truncates the list
to `k`, so it always keeps exactly `k`; which of several tied candidates survive
depends on the sort. Its default is `top_k = 40`. Hugging Face's `TopKLogitsWarper`
clamps `k` to the vocabulary size and removes `scores < topk(scores, k).values[..., -1]`,
the threshold form, and rejects a non-positive `top_k`; vLLM's PyTorch path masks
`logits < k-th largest value` in the same way. mlx-lm's `apply_top_k` uses
`mx.argpartition(-logprobs, kth=top_k - 1)` and masks everything after position
`top_k`; its `make_sampler` treats `top_k = 0` as "off". None of them define the order
of ties, which is one reason the same seed gives different text in different engines.
