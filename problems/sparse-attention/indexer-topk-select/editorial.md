## The idea

A sparse attention layer splits its work into two stages. First, the indexer cheaply decides which earlier tokens matter for each query. Second, standard attention computes the weighted value sum over only those selected tokens. This problem implements the first stage.

The index score strips standard attention down to four lightweight design choices:

- **One key per token.** `k_idx[s]` has no head axis. All heads share the same key vector, so caching indexer keys costs $D$ numbers per token rather than $H \cdot D$.
- **A ReLU per head.** Each head computes a dot product and keeps only positive values. A head either votes for a token or abstains rather than voting negatively.
- **A signed, query-dependent weight per head.** The weight $w_{t,h}$ comes from query token $t$ and controls how much head $h$ counts. A negative weight turns a head's positive votes into penalties.
- **No softmax.** Only the relative ordering of scores within a row matters. Skipping normalization means a key's score never depends on other keys, so cached keys do not need to be rescored as the sequence grows.

The two scale factors, $D^{-1/2}$ on the dot product and $H^{-1/2}$ on the weights, do not change which keys win because $\operatorname{relu}(c\,x) = c \operatorname{relu}(x)$ for any $c > 0$. They remain part of the specification because the shipped model includes them, and any downstream check or threshold on raw scores depends on their exact values.

Three details determine whether `topk_mask` is exact:

1. **Mask before selecting.** The $k$ slots belong exclusively to visible entries. Running top-k over all scores and masking afterward wastes slots on high-scoring invisible entries, leaving fewer than $k$ visible selections. Replace invisible scores with $-\infty$ before ranking. Do not use `0`, because visible scores can be negative when dominant heads have negative weights.
2. **Intersect with `visible` afterward.** A row with fewer than $k$ visible entries fills its remaining top-$k$ slots with invisible entries at $-\infty$. Intersecting the result with `visible` removes them.
3. **Break ties deterministically.** The per-head ReLU makes exact ties common, since every key that no head votes for scores `0.0` (or `-0.0`). Standard top-k routines return an arbitrary subset among tied entries. Guaranteeing lowest-index-first tie-breaking requires a stable sort.

## Reference solution (PyTorch)

```python
def index_scores(q_idx, k_idx, w):
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores, visible, k):
    masked = torch.where(visible, scores, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return (rank < k) & visible
```

In `topk_mask`, `order[t]` lists the key indices of row `t` from best to worst. The second `argsort` inverts that permutation so `rank[t, s]` gives the 0-based rank of key `s`. In JAX and MLX, sorting is stable by default and has no `descending` flag, so sorting `-masked` in ascending order produces the same ranking and tie order. Comparing `rank < k` works without clamping even when $k > S$.

## Common mistakes

- **Applying ReLU after the sum.** Computing `relu(sum_h w * dot)` allows negative dot products to subtract from the total and clips negative sums to zero. In the worked example, this changes the score of key 1 for query 0.
- **Using absolute value instead of ReLU.** A strong negative dot product then counts the same as a strong positive match.
- **Applying ReLU to the weights or omitting weights.** Both prevent negative head weights from lowering a score.
- **Dropping a scale factor.** Omitting $D^{-1/2}$ or $H^{-1/2}$ scales every score by a constant factor. Selections do not change, so this bug slips past tests that check only the mask.
- **Running top-k before masking.** High-scoring invisible entries consume slots that belong to visible entries.
- **Masking invisible entries with `0`.** Invisible entries then outrank visible entries with negative scores.
- **Skipping the final `& visible` intersection.** Rows with fewer than $k$ visible entries end up selecting invisible entries sitting at $-\infty$.
- **Using a value threshold instead of a rank count.** Checking `masked >= kth_value` keeps every entry tied with the $k$-th score and can select more than $k$ entries.
- **Leaving ties to the library or breaking toward higher indices.** `torch.topk` makes no guarantee on tie order and does not return the lowest index on the test inputs.
- **Adding a tiny index offset to break ties.** Subtracting `1e-6 * arange(S)` fails at both extremes. Near `1000`, `1e-6` is below float32 precision and breaks no tie, while between scores `1e-7` apart it reverses the true order. Tie-breaking must come from the sort comparison, not arithmetic on scores.
- **Masking in place.** Calling `scores.masked_fill_(~visible, -inf)` mutates the caller's tensor.

## Complexity

`index_scores` performs $O(T\,S\,H\,D)$ multiply-adds and uses $O(T\,H\,S)$ memory for the per-head score tensor. Although still quadratic in sequence length like dense attention, its width $H \cdot D$ (4096 in GLM-5.3) is much smaller than full attention, and later layers can reuse a selection (`shared-index-layers`). Sorting in `topk_mask` takes $O(T\,S \log S)$ time. Partial selection runs in $O(T\,S)$ time, but leaves tie order unspecified.

## In the wild

Checked on 2026-11-21.

- **The formula.** "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models" (arXiv 2512.02556), Eq. 1, defines $I_{t,s} = \sum_j w^I_{t,j} \cdot \operatorname{ReLU}(q^I_{t,j} \cdot k^I_s)$ and notes that ReLU was chosen "for throughput consideration". Eq. 2 attends over the entries with the top-k scores. The paper omits the two scale factors.
- **The code.** `GlmMoeDsaIndexer.forward` in `transformers` 5.23.0 (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) computes `relu(q @ k^T * head_dim ** -0.5)`, multiplies by `weights_proj(x) * n_heads ** -0.5`, sums over heads, fills masked positions with `-inf`, and returns `topk(min(index_topk, S))` indices. `GlmMoeDsaAttention` scatters those indices into a mask and intersects it with the causal mask. The public test "GlmMoeDsaIndexer" runs that module with the indexer dimensions of `zai-org/GLM-5.3` (`index_n_heads` 32, `index_head_dim` 128, `qk_rope_head_dim` 64 from its `config.json`, with `index_topk` set to 4 in the fixture instead of 2048) and compares both your scores and your mask against it. `fixtures/make.py` regenerates the fixture.

Several production details are omitted from this problem:

- **Input projections.** In the shipped model, queries come from projecting (`wq_b`) the low-rank query residual of the main attention, keys are `LayerNorm(wk(x))` with eps `1e-6`, weights are `weights_proj(x)`, and rotary embeddings are applied to the first `qk_rope_head_dim` channels of queries and keys. Here `q_idx`, `k_idx`, and `w` arrive already projected and rotated.
- **Batching and caching.** The batch axis and the indexer key cache (`indexer-cache-decode`) are omitted.
- **Return type.** The shipped indexer returns top-k indices, and the attention layer builds the boolean mask. `topk_mask` returns that final mask directly.
- **Tie-breaking.** The shipped code uses `torch.topk`, which leaves tie order unspecified. Lowest-index-first is this problem's rule rather than the model's. With 32 heads, an exact tie between visible entries requires every head to abstain on both, so ties are rare in practice. The fixture has no ties at its cutoffs, and `fixtures/make.py` asserts this.
