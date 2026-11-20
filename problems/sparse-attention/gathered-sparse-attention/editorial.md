## The idea

Sparse attention is ordinary attention with a different set of keys for every query.
Dense code has one key tensor `(S, H, d)` that all queries share; here query $t$ has
its own small key tensor, the rows `idx[t]` of the store. Gathering with the whole
index tensor gives `(T, k, H, d)`: a private set of $k$ keys and values per query.
From there on the formulas are those of attention with the key axis renamed from
"token" to "slot", and the score tensor shrinks from `(T, H, S)` to `(T, H, k)`.

That is the entire saving, and it is why the store counts. Scoring all $S$ keys and
adding a mask built from `idx` gives the same output to the last bit of the
tolerance. It also does $T \cdot S$ dot products per head and reads $S$ rows. The
count separates the two.

The index tensor is rectangular and the selections are not, so some slots are
padding, marked `-1`. Padding has to be handled twice, for two different reasons:

1. **It must not be read.** `-1` is a valid index in every framework: it is the last
   row. Passing `valid=idx >= 0` to `gather` skips those slots; they are not counted
   and come back as zero rows.
2. **It must not be attended to.** A zero key has the score 0, and $e^0 = 1$ is a
   large weight next to real scores that may be negative. The score of a padded slot
   has to be $-\infty$, as for any masked position.

The last corner is the query with no used slot at all. Its scores are all $-\infty$
and the softmax divides zero by zero. The specification says such a query returns
zeros, so the NaN has to be replaced, which the same `valid` mask does.

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

The second `where` changes only the rows that are entirely padding: everywhere else
the weight of a padded slot is already $e^{-\infty} = 0$.

## Common mistakes

- **The sentinel as an index.** `store.gather(idx)` reads the last token for every
  padded slot and then attends to it. Nothing raises. In the worked example the last
  key matches query 1 as well as its real selection does, and the output moves
  halfway to the last token's value.
- **Read, then mask.** Clamping the padding to row 0 (`idx.clamp(min=0)`) and masking
  afterward gives the right output and reads one extra row per padded slot. On real
  shapes early queries are mostly padding.
- **Padding scored 0.** Skipping the read is not enough. Without the $-\infty$ every
  padded slot takes the weight of a key orthogonal to the query.
- **Masking after the softmax.** Zeroing the weights of padded slots afterward
  leaves the real weights summing to less than 1: the padding was in the denominator.
- **Everything read, then masked.** The dense formulation: correct numbers, $S$ rows.
- **Scale $k^{-1/2}$.** The scale belongs to the dot product, whose variance grows
  with the width $d$. It has nothing to do with how many keys there are.
- **One selection for all queries.** Broadcasting `idx[0]` is what dense attention
  does with its single key tensor. Here every query has its own row.
- **NaN for an empty row.** The softmax over no entries.
- **Modifying `idx`.** `idx[idx < 0] = 0` is a convenient way to make the gather
  safe, and it destroys the caller's selection. A shared layer attends with the same
  indices later (`shared-index-layers`).
- **Going around the counter:** `store.reference()`, the private arrays, a twin from
  `with_framework`, or resetting the counter after reading everything. The harness
  rejects each; those members exist for the tests.

## Complexity

$O(T\,k\,H\,d)$ time and memory, and $\sum_t |J_t| \le T k$ rows read, independent of
$S$. Dense attention is $O(T\,S\,H\,d)$. The price is the gather: `(T, k, H, d)` is a
copy, where dense attention reads its keys in place. A real kernel reads the selected
rows straight from the cache without materializing them per query.

## In the wild

Checked on 2026-11-21 in `transformers` 5.23.0.

- **The dense path.** `GlmMoeDsaAttention.forward`
  (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) under the `eager` and `sdpa`
  attention implementations scatters the indexer's top-k indices into a boolean mask
  over all keys, intersects it with the causal mask, and runs attention over every
  key: the "everything read, then masked" form. It is the reference for the numbers,
  not for the cost.
- **The gathered path.** For any other attention implementation the same method
  passes the indices on (`indices=sparse_indices`), with the comment that they are
  "consumed by flash_mla_with_kvcache": a kernel that reads only the selected rows.
  That kernel is not part of `transformers` and was not read.
- **The sentinel.** `DeepseekV4Indexer.forward`
  (`models/deepseek_v4/modeling_deepseek_v4.py`) replaces picks that a query may not
  see yet with `-1`, and `DeepseekV4CSACompressor.forward` computes
  `valid = top_k_indices >= 0` and redirects the invalid ones before it scatters
  them. GLM's indexer returns no sentinel: its rows always hold `min(k, S)` real
  indices, and the entries a query may not see are removed by the causal mask.

What this problem leaves out: the batch axis; the latent form of the keys and values
in GLM-5.3 and DeepSeek-V3.2 (one compressed vector per token that is expanded into
per-head keys and values after the gather; here the store holds per-head rows); the
rotary position embedding; and the sliding-window keys that DeepSeek-V4 concatenates
with its gathered entries, which is why a query with no valid pick is harmless there.
In this problem such a query returns zeros.
