## The idea

Selecting the `k` largest entries is a question about ranks, not values. A value threshold checks which entries are at least as large as the k-th value, which disagrees with the rank definition whenever values tie at the boundary.

To compute ranks, sort each row in descending order with a stable sort. Stability enforces the tie-breaking rule because equal logits stay in their original index order, giving lower indices better ranks. This sort returns `order`, which maps each rank to its token index. Inverting `order` produces `rank`, which maps each token index to its unique rank in $0, \ldots, V - 1$. Because every rank is distinct, `rank < k` is `True` for exactly `k` entries per row.

Masked entries require no special case. Since `-inf` sorts after every finite value, a masked token only falls in the top `k` when `k` exceeds the number of finite entries, and keeping it copies `-inf`.

## Reference solution (PyTorch)

```python
def top_k_filter(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return torch.where(rank < k, logits, float("-inf"))
```

`argsort(order)` inverts the permutation. Since `order[b, r] = v` means token `v` has rank `r`, sorting the token indices `order[b, :]` in ascending order places `r` at slot `v`. If you want to avoid the second sort, you can scatter into a boolean mask instead (as in `top-p-filtering`) using `keep = zeros(B, V, bool).scatter(-1, order[:, :k], True)`.

In JAX and MLX, `argsort` is already stable and has no `descending` flag. Sorting `-logits` produces the exact same ranking because negation preserves equality, so the stable sort still orders tied values by lowest index.

The early return handles both edge conventions. Without checking `k <= 0`, the condition `rank < 0` is `False` everywhere and masks the entire batch.

## Common mistakes

- **Using a value threshold.** Checking `logits >= kth_largest` keeps every entry tied with the k-th value. If all logits are equal and `k = 1`, it keeps the entire row. Hugging Face's `TopKLogitsWarper` uses this formulation, which is fine when an approximate count is acceptable, but it breaks downstream code that expects at most `k` candidates.
- **Treating `k = 0` literally.** Masking every token leaves a row of `-inf`, whose softmax is NaN. Clients send `0` (often the default value) to disable top-k filtering.
- **Not handling `k > V`.** Calling `topk(logits, k)`, indexing `sorted[:, k - 1]`, or calling `kthvalue` raises an error when `k` exceeds the vocabulary size. This happens in production whenever a default `k` meets a small vocabulary or a pre-filtered candidate set.
- **Reversing ties.** Sorting in ascending order and reversing the result puts logits in descending order, but it also reverses the index order of tied entries. Use a descending stable sort, or sort `-logits` in ascending order.
- **Off-by-one boundary.** Writing `rank <= k` or slicing `order[:, : k + 1]` keeps `k + 1` tokens instead of `k`.
- **Masking with a large negative constant.** Filling removed slots with `-1e9` or `finfo.min` instead of `-inf` leaves those tokens alive. Downstream filters or "keep at least `n`" rules still count them as candidates, and checks for `== -inf` treat them as unmasked.
- **Ranking across the whole batch.** Flattening `logits` before sorting (or reducing the threshold to a single scalar) keeps the `k` largest logits globally rather than per row, so rows with larger logits keep many tokens while other rows keep none.

## Complexity

$O(B \cdot V \log V)$ time for sorting and $O(B \cdot V)$ memory. Because only the top `k` elements need to be separated rather than fully ordered, production engines often use partial selection (`topk` or `argpartition`) in $O(V + k \log k)$ time per row, at the cost of unspecified tie-breaking.

## In the wild

llama.cpp's `llama_sampler_top_k_impl` returns immediately when `k <= 0`, clamps `k` to the number of candidates, partially sorts them, and truncates to `k` (defaulting to `top_k = 40`). It always keeps exactly `k` candidates, though tie-breaking depends on the sort.

Hugging Face's `TopKLogitsWarper` clamps `k` to the vocabulary size, rejects non-positive `top_k`, and uses the threshold form (`scores < topk(scores, k).values[..., -1]`). vLLM's PyTorch path masks `logits < k-th largest value` the same way. mlx-lm's `apply_top_k` uses `mx.argpartition(-logprobs, kth=top_k - 1)` and masks positions from `top_k` onward, while `make_sampler` treats `top_k = 0` as disabled. None of these engines specify tie order, which is one reason the same random seed can produce different text across engines.
