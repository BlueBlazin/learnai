## The idea

Sparse attention is ordinary attention where each query has its own small set of keys and values. Dense attention shares one `(S, H, d)` key tensor across all queries. Here, gathering with `idx` produces `(T, k, H, d)` tensors, giving each query $t$ its own $k$ keys and values. From there, standard attention runs over the slot axis instead of the token axis, shrinking the score tensor from `(T, H, S)` to `(T, H, k)`.

Scoring all $S$ keys and masking out unselected tokens gives the same numerical result, but performs $T \cdot S$ dot products per head and reads all $S$ rows. The store's read counter enforces the gathered path.

Because `idx` is rectangular while each query's selection size can vary, unused slots are padded with `-1`. Padding must be handled in two separate places:

1. **Skip reading it.** `-1` is a valid Python index that reads the last row of the store. Passing `valid=idx >= 0` to `store.gather` skips padded slots so they are neither read nor counted, returning zero rows in their place.
2. **Mask it before softmax.** A zero key produces a dot product of `0`, and $e^0 = 1$ carries significant weight whenever real scores are negative. Padded slots must have their scores replaced with $-\infty$ before softmax.

One edge case remains when a query has no used slots. All of its masked scores are $-\infty$, causing softmax to divide zero by zero and produce `NaN`. Applying the same `valid` mask to the softmax weights replaces those `NaN` values with `0.0`, returning an all-zero output as required.

## Reference solution (PyTorch)

```python
def sparse_attend(q, store, idx):
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)  # (T, k, H, d)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
```

The second `torch.where` only changes rows that are completely padding. On every other row, padded slots already have weight $e^{-\infty} = 0$.

## Common mistakes

- **Using the sentinel as an index.** Calling `store.gather(idx)` without `valid` silently reads the last token for every `-1` slot and attends to it. In the worked example, the last key matches query 1 as strongly as its true selection, pulling the output halfway toward the last token's value.
- **Reading first, then masking.** Clamping padding with `idx.clamp(min=0)` and masking afterward gives the right numbers, but wastes one row read per padded slot. In real sequences, early queries are mostly padding.
- **Leaving padded scores at `0`.** Skipping the read is not enough. Without replacing padded scores with $-\infty$, every padded slot receives the same weight as a key orthogonal to the query.
- **Masking after softmax.** Zeroing padded weights after softmax leaves the remaining real weights summing to less than 1 because the padded slots stayed in the denominator.
- **Reading everything and masking.** The dense formulation returns the right values but reads all $S$ rows and fails the cost check.
- **Scaling by $k^{-1/2}$.** The scale factor normalizes the dot product, whose variance grows with head width $d$, not with the number of slots $k$.
- **Sharing one selection across all queries.** Broadcasting `idx[0]` mirrors dense attention's shared key tensor, whereas each query here has its own row of `idx`.
- **Returning `NaN` on an empty row.** Softmax over all $-\infty$ entries produces `NaN` unless replaced with zeros.
- **Modifying `idx` in place.** Overwriting `idx[idx < 0] = 0` mutates the caller's selection tensor, which breaks downstream layers that reuse the same indices (`shared-index-layers`).
- **Bypassing the counter.** Calling `store.reference()`, touching private arrays, using `with_framework`, or resetting counters is blocked by the harness. Those helpers exist only for tests.

## Complexity

Time and memory complexity are $O(T\,k\,H\,d)$, and the number of rows read is $\sum_t |J_t| \le T k$, independent of $S$. Dense attention costs $O(T\,S\,H\,d)$. Materializing the `(T, k, H, d)` gathered tensors does copy data, whereas a fused kernel reads selected rows directly from the KV cache without materializing a per-query copy.

## In the wild

Checked on 2026-11-21 in `transformers` 5.23.0.

- **The dense path.** Under the `eager` and `sdpa` attention backends, `GlmMoeDsaAttention.forward` (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) scatters the indexer's top-k indices into a boolean mask over all keys, intersects it with the causal mask, and runs dense attention. This matches the numbers but not the sparse cost.
- **The gathered path.** For other backends, the same method forwards `indices=sparse_indices` with a comment noting they are "consumed by flash_mla_with_kvcache", a kernel that reads only the selected rows (not part of `transformers` and not inspected here).
- **The sentinel.** `DeepseekV4Indexer.forward` (`models/deepseek_v4/modeling_deepseek_v4.py`) replaces future picks with `-1`, and `DeepseekV4CSACompressor.forward` computes `valid = top_k_indices >= 0` to redirect invalid entries before scattering. GLM's indexer uses no sentinel. Its rows always hold `min(k, S)` real indices and rely on the causal mask to remove future entries.

This problem omits the batch axis, rotary position embeddings, and the latent KV representation used in GLM-5.3 and DeepSeek-V3.2 (where one compressed vector per token is gathered and expanded into per-head keys and values afterward, rather than storing per-head rows directly). It also omits the sliding-window keys that DeepSeek-V4 concatenates with gathered entries, which prevent empty selections in production. Here, an empty selection returns zeros.
