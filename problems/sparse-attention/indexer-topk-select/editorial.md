## The idea

A sparse attention layer answers two questions in two places. *Which tokens matter to
this query?* is answered by the indexer, for every earlier token, cheaply. *What do
they contribute?* is answered by ordinary attention, over the chosen tokens only. This
problem is the first question.

The index score is an attention score stripped down until it is cheap:

- **One key per token.** `k_idx[s]` has no head axis. All heads share it, so the keys
  the indexer caches cost $D$ numbers per token, not $H \cdot D$.
- **A ReLU per head.** Each head computes a dot product and keeps it only if it is
  positive. A head either votes for a token or abstains; it cannot vote against one.
- **A signed, query-dependent weight per head.** $w_{t,h}$ comes from the query token.
  It decides how much each head's votes count for this query, and a negative weight
  turns a head's votes into penalties.
- **No softmax.** Only the order of the scores within a row is used. Nothing
  normalizes them, so the score of one key does not depend on the other keys, and a
  cached key never has to be rescored when the sequence grows.

The two constants, $D^{-1/2}$ on the dot product and $H^{-1/2}$ on the weights, do not
change any selection ($\operatorname{relu}(c\,x) = c \operatorname{relu}(x)$ for
$c > 0$). They are part of the function all the same: they are in the shipped code,
and anything that compares or thresholds scores depends on them.

Selection is where the definitions hide. Three choices decide the result:

1. **Mask, then select.** The $k$ slots belong to visible entries. If the top-k runs
   over all scores and the mask is applied afterward, a row whose best-scoring
   entries are invisible spends its slots on them and is left with fewer than $k$.
   The fix is to make invisible entries unable to win, with $-\infty$. Not with 0:
   scores are negative whenever the dominant heads have negative weights.
2. **Intersect afterward as well.** A row with fewer than $k$ visible entries fills
   its remaining slots with entries at $-\infty$. They have to be removed.
3. **Ties.** The ReLU makes exact ties common: every key that no head votes for
   scores exactly 0 (with either sign of zero, depending on the weights). A top-k
   routine returns *some* $k$ of the best entries; which of several equal ones is
   unspecified. This problem fixes the order (lowest index first), which needs a
   stable sort.

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

`order[t]` lists the entries of row `t` from best to worst, and the second `argsort`
inverts that permutation: `rank[t, s]` is the place of entry `s`. In JAX and MLX the
sort is stable by default and has no `descending` argument; sorting `-masked` in
ascending order gives the same ranking, ties included. `rank < k` needs no clamp for
$k > S$.

## Common mistakes

- **ReLU after the sum.** `relu(sum_h w * dot)` lets a head with a negative dot
  product subtract from the score, and clips negative totals to 0. In the example it
  changes the score of key 1 for query 0.
- **Absolute value for ReLU.** A strong mismatch then counts like a strong match.
- **ReLU on the weights too,** or the weights left out. Both remove the only way a
  head can lower a score.
- **A scale dropped.** Without $D^{-1/2}$ or $H^{-1/2}$ every score is off by a
  constant factor. No selection changes, which is why this survives testing against
  selections alone.
- **Top-k before the mask.** The row loses visible entries to invisible ones.
- **Masking with 0.** The same loss, on rows whose visible scores are negative.
- **No intersection.** Rows with fewer than $k$ visible entries select invisible ones.
- **A threshold instead of a count.** `masked >= kth_value` keeps every entry tied
  with the $k$-th and can select more than $k$.
- **Ties left to the library,** or resolved toward the highest index. `torch.topk`
  gives no guarantee, and on the inputs of the tests it does not return the lowest
  index.
- **A small offset per index** to make `topk` prefer low indices,
  `masked - 1e-6 * arange(S)`. Near 1000 the offset is below float32 resolution and
  breaks no tie; between scores `1e-7` apart it reverses a real order. A tie rule
  has to be a comparison, not arithmetic on the scores.
- **Masking in place.** `scores.masked_fill_(~visible, -inf)` destroys the caller's
  scores.

## Complexity

`index_scores` costs $O(T\,S\,H\,D)$ multiply-adds and $O(T\,H\,S)$ memory for the
per-head tensor. That is quadratic in the sequence length, like attention, but with a
small $H \cdot D$ (4096 for GLM-5.3) in place of the full attention width, and it is
the reason later layers reuse a selection (`shared-index-layers`). `topk_mask` by
sorting costs $O(T\,S \log S)$; a partial selection is $O(T\,S)$ but leaves ties open.

## In the wild

Checked on 2026-11-21.

- **The formula.** "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
  (arXiv 2512.02556), Eq. 1, defines
  $I_{t,s} = \sum_j w^I_{t,j} \cdot \operatorname{ReLU}(q^I_{t,j} \cdot k^I_s)$ and says
  ReLU was chosen "for throughput consideration"; Eq. 2 attends over the entries with
  the top-k scores. The two scale factors are not in the paper.
- **The code.** `GlmMoeDsaIndexer.forward` in `transformers` 5.23.0
  (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) computes
  `relu(q @ k^T * head_dim ** -0.5)`, multiplies by
  `weights_proj(x) * n_heads ** -0.5`, sums over heads, fills the masked positions
  with `-inf`, and returns `topk(min(index_topk, S))` indices. `GlmMoeDsaAttention`
  scatters those indices into a mask and intersects it with the causal mask. The
  public test "GlmMoeDsaIndexer" runs that module with the indexer sizes of
  `zai-org/GLM-5.3` (`index_n_heads` 32, `index_head_dim` 128, `qk_rope_head_dim` 64,
  from its `config.json`; `index_topk` is 2048 there and 4 in the fixture) and
  compares your scores and your mask with its own. `fixtures/make.py` regenerates it.

What this problem leaves out of the shipped indexer:

- **Where the inputs come from.** The shipped queries are a projection (`wq_b`) of the
  low-rank query residual of the main attention; the key is
  `LayerNorm(wk(x))` with eps `1e-6`; the weights are `weights_proj(x)`. A rotary
  embedding is applied to the first `qk_rope_head_dim` channels of queries and keys.
  Here `q_idx`, `k_idx`, and `w` arrive computed (in the fixture: after the rotation).
- **The batch axis,** and the indexer key cache (`indexer-cache-decode`).
- **The return type.** The shipped indexer returns indices, and the mask is built in
  the attention layer. `topk_mask` returns that final mask.
- **Ties.** The shipped code calls `torch.topk`, whose order among equal scores is
  not specified. "Lowest index first" is this problem's rule, not the model's. With
  32 heads an exact tie between visible entries needs every head to abstain on both,
  so it is rare in the shipped model; the fixture has none at its cuts
  (`fixtures/make.py` asserts it).
