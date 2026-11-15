## The idea

The policy has two parts, and each answers a different question.

**Which old tokens matter?** The ones the model keeps coming back to. A token's
accumulated score is the sum of the attention it has received from every later query
(and itself), which for a prompt is a column sum of the attention matrix and for a
decode step is one more row added to the totals. The H2O paper's observation is that
these totals are heavily skewed: a few tokens collect most of the mass, and the
score computed online, from past steps only, selects nearly the same tokens as a
score that could see the future.

**Why protect the newest tokens?** Because the score is a sum over time, and a token
that has existed for one step has had one chance to collect anything. Ranking by
score alone evicts the newest entries first. The recent window gives a token time to
prove itself, and it is also where the next query looks most. The paper splits the
budget evenly between the two parts; here `recent` is a parameter.

Everything else is alignment. After the first eviction the cache of a head is an
arbitrary increasing subsequence of the tokens, different in every head. So the
function never reasons about token numbers. It works with positions in the cache it
was handed, produces one index tensor `kept` of shape `(B, H, N)`, and applies that
same tensor to keys, values, and scores. As long as `kept` is increasing, sequence
order survives every step by induction, and with it the meaning of "the last
`recent` positions".

For one token per step at a full cache ($S = \text{budget} + 1$) the rule evicts
exactly one entry per head, the lowest-scoring one outside the window, which is the
greedy step of the paper's Algorithm 1. Handling a chunk of `T` tokens with the same
code is what cuts a long prompt down to the budget after the prefill.

## Reference solution (PyTorch)

```python
def h2o_step(cache, scores, attn_weights, budget, recent):
    k, v = cache
    b, h, t, s = attn_weights.shape
    received = attn_weights.sum(dim=2)
    if scores is None:
        scores = received
    else:
        scores = torch.cat((scores, torch.zeros(b, h, t)), dim=2) + received
    if s <= budget:
        kept = torch.arange(s).expand(b, h, s)
    else:
        old = s - recent
        ranked = torch.argsort(scores[:, :, :old], dim=-1, descending=True, stable=True)
        heavy = torch.sort(ranked[:, :, : budget - recent], dim=-1).values
        window = torch.arange(old, s).expand(b, h, recent)
        kept = torch.cat((heavy, window), dim=-1)
    index = kept[..., None].expand(-1, -1, -1, k.shape[-1])
    cache = (torch.gather(k, 2, index), torch.gather(v, 2, index))
    return cache, torch.gather(scores, 2, kept), kept
```

The stable descending sort is the tie rule: equal scores keep their original order,
so the older position ranks first. `torch.topk` would be shorter and makes no
promise about ties. In JAX and MLX `argsort` is stable, so the ranking is
`argsort(-scores)`, and the gathers are `take_along_axis`.

`recent = budget` and `recent = 0` need no special cases: the slices
`ranked[:, :, :0]` and `arange(S, S)` are empty and concatenate to nothing.

## Common mistakes

- **Ranking by the latest attention.** The newest row is one sample. A token that
  the model reads every few steps has a high total and, at most steps, a low latest
  weight; it is exactly the token this policy exists to keep.
- **Scores not gathered.** Returning the first `N` scores, or the full tensor, leaves
  them attached to positions, not tokens. After one eviction every score to the
  right of the hole belongs to the wrong entry, and every later decision is made on
  shifted data.
- **Order not preserved.** `topk` returns positions by score. Gathering in that
  order scrambles the sequence; the window is then no longer "the last `recent`
  positions" on the next call, and anything that assumes order (a causal mask over a
  multi-token step, position-dependent logic) breaks.
- **No recent window.** Pure top-`budget` evicts the token that was just appended.
- **One ranking for all heads.** Summing the scores over heads keeps the same
  positions everywhere. It is simpler to store, and it discards what the per-head
  scores know: heads specialize, and one head's heavy hitter is another's noise.
- **Ties broken the other way.** A reversed or unstable sort keeps the newer of two
  equal tokens. Ties are common at the start, when many scores are still 0.
- **Only the last query of a chunk.** For a prefill, `attn_weights[:, :, -1]` is one
  row of the matrix; the score is the sum of all `T` rows.

## Complexity

Per step, the accumulation is $O(B\,H\,T\,S)$, the sort $O(B\,H\,S \log S)$, and the
gathers $O(B\,H\,N\,D)$, which copy the whole kept cache. A production implementation
avoids both the sort and the copy for single-token steps: only one entry leaves, so
it finds the minimum outside the window and overwrites that slot. The paper's
implementation notes describe a preallocated buffer whose last entries are a circular
queue for the recent tokens.

Memory is `budget` entries per head and layer whatever the sequence length, plus one
float per entry for the score.

## In the wild

The method is from "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models" (arXiv 2306.14048, NeurIPS 2023). The paper defines heavy
hitters by the accumulated attention score, retains "a balance of recent and H2
tokens", assigns the budget evenly to the two groups in its experiments, and evicts
per head: its implementation notes say that at each decoding step "one heavy hitter
will be evicted for each head". One property of the policy is worth knowing: an
evicted token is gone for good. The policy cannot bring it back if the model would
have attended to it later, which is why the window and the budget are tuned per
task. Related policies choose the survivors differently (a sliding window with
attention sinks, see `sliding-window-cache`; a one-shot selection from the prompt's
last tokens, see `snapkv-compress`).
