Compose the samplers from earlier problems into a full pipeline that turns a batch of
logits into sampled token IDs. You will apply penalties, top-k, top-p, min-p,
temperature scaling, and inverse-CDF sampling in sequence. Because each stage operates
on the output of the previous stage, the stage order determines which tokens survive.

![This problem is the whole chain, from a row of logits to one token.](assets/map.svg)

![The dashed outline is the model's distribution and a filled bar is the probability the chain samples from. Switch the order with the same settings and the kept set changes.](interactive:sampler-order)

## Specification

```python
def chain_logits(logits, cfg, history): ...     # the logits after stages 1 to 5
def sample_chain(logits, cfg, history, u): ...  # one token per row
```

The batch has `B` rows over a vocabulary of size `V`, with a history of `T` previously
generated tokens per row ($T \ge 1$).

- `logits` holds raw scores for each row. Entries may already be `-inf` (for example,
  from a grammar mask), and every row has at least one finite entry.
- `history` holds token IDs in $[0, V)$ that each row has produced so far. Every token
  in `history` counts toward the penalties.
- `u` holds one uniform number per row in $[0, 1)$, used for inverse-CDF sampling.
- `cfg` is a dict with exactly these seven keys:

| key | type | range | value that switches the stage off |
|---|---|---|---|
| `"repetition_penalty"` | float | $\ge 1$ | `1.0` |
| `"frequency_penalty"` | float | $\ge 0$ | `0.0` |
| `"presence_penalty"` | float | $\ge 0$ | `0.0` |
| `"top_k"` | int | $\ge 0$ | `0`, or any value $\ge V$ |
| `"top_p"` | float | $0 < p \le 1$ | `1.0` |
| `"min_p"` | float | $0 \le m < 1$ | `0.0` |
| `"temperature"` | float | $> 0$ | `1.0` |

The stages run in the default order of llama.cpp. Each row is processed independently.
Let $x$ denote one row of logits. "Removing" a token means setting its logit to `-inf`.

**1. Penalties.** Let $c_v$ be the number of times token $v$ occurs in the row's
`history`. For every token with $c_v > 0$, apply the repetition penalty $\theta$ first,
then subtract the frequency penalty $\alpha_f$ and presence penalty $\alpha_p$:

$$
x_v \leftarrow
\begin{cases}
x_v / \theta & \text{if } x_v > 0, \\
x_v \cdot \theta & \text{if } x_v \le 0,
\end{cases}
\qquad\text{then}\qquad
x_v \leftarrow x_v - c_v\,\alpha_f - \alpha_p.
$$

Tokens that do not appear in `history` are unchanged. The repetition and presence
penalties apply once per distinct token, while the frequency penalty applies once per
occurrence.

**2. Top-k.** Keep the `top_k` largest logits and remove the rest.

**3. Top-p.** Compute the softmax of the current logits. Because removed tokens have
probability `0`, this distribution is **renormalized over the survivors of top-k**. Rank
tokens by decreasing probability and keep a token when the cumulative probability mass
ranked strictly before it is below `top_p`. The token that pushes the cumulative mass to
or past `top_p` is kept.

**4. Min-p.** Compute the softmax of the current logits again and remove every token
whose probability is below `min_p` times the largest probability in the row.

**5. Temperature.** Divide every logit by `temperature`. Removed tokens remain `-inf`.

`chain_logits` returns the result of stage 5.

**6. Sample.** `sample_chain` computes
$\pi = \operatorname{softmax}(\texttt{chain\_logits}(\ldots))$ and draws one token per
row by inverse-CDF sampling using that row's uniform value $u$. Let
$c_k = \sum_{i \le k} \pi_i$ be the cumulative probability sum. Return the smallest
index $k$ satisfying $c_k > u \cdot c_{V-1}$. When `u` is uniform on $[0, 1)$, token $k$
is drawn with probability $\pi_k$, and a removed token is never drawn.

Any stage switched off by `cfg` must leave the logits unchanged. In particular,
`top_p = 1.0` disables the top-p stage completely, so no token may be removed regardless
of how small its probability is.

The kit provides the three filters for stages 2 to 4 and a softmax helper:

```python
from learnai.kit import ops

ops.top_k_filter(logits, k)       # k <= 0 or k >= V: unchanged
ops.top_p_filter(logits, p)
ops.min_p_filter(logits, min_p)   # min_p = 0: unchanged
ops.softmax(logits)               # -inf entries get probability 0
```

Each helper operates along the last axis and returns logits where removed entries are
`-inf`, kept entries are unchanged, and existing `-inf` entries stay `-inf`. Equal
logits are ranked by token index, lowest first. You may use these helpers or write your
own. Any implementation that matches the six stage definitions is accepted.

Randomness is an input. Do not draw random numbers, and do not call your framework's
categorical or multinomial sampler.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `history` | `(B, T)` | integer |
| `u` | `(B,)` | float32 |
| result of `chain_logits` | `(B, V)` | float32 |
| result of `sample_chain` | `(B,)` | integer |

## Example

```python
logits  = [[3.0, 2.5, 2.0, 1.0, 0.0, -1.0]]
history = [[0, 0, 5]]
cfg = {"repetition_penalty": 1.2, "frequency_penalty": 0.5, "presence_penalty": 0.25,
       "top_k": 4, "top_p": 0.8, "min_p": 0.3, "temperature": 2.0}

chain_logits(logits, cfg, history)         # [[-inf, 1.25, 1.0, -inf, -inf, -inf]]
sample_chain(logits, cfg, history, [0.3])  # [1]
sample_chain(logits, cfg, history, [0.7])  # [2]
```

| stage | logits afterwards | softmax afterwards |
|---|---|---|
| penalties | `[1.25, 2.5, 2.0, 1.0, 0.0, -1.95]` | |
| top-k, 4 | `[1.25, 2.5, 2.0, 1.0, -inf, -inf]` | `[0.135, 0.473, 0.287, 0.105, 0, 0]` |
| top-p, 0.8 | `[1.25, 2.5, 2.0, -inf, -inf, -inf]` | `[0.151, 0.528, 0.320, 0, 0, 0]` |
| min-p, 0.3 | `[-inf, 2.5, 2.0, -inf, -inf, -inf]` | `[0, 0.622, 0.378, 0, 0, 0]` |
| temperature, 2.0 | `[-inf, 1.25, 1.0, -inf, -inf, -inf]` | `[0, 0.562, 0.438, 0, 0, 0]` |

In stage 1, token 0 occurred twice and has a positive logit, so it becomes
$3.0 / 1.2 - 2 \cdot 0.5 - 0.25 = 1.25$. Token 5 occurred once and has a negative logit,
so it becomes $-1.0 \cdot 1.2 - 0.5 - 0.25 = -1.95$. Token 0 drops from first place to
third.

In stage 2, top-k keeps the four largest logits. In stage 3, the sorted survivor
probabilities are `0.473`, `0.287`, `0.135`, and `0.105`, with preceding cumulative
masses `0`, `0.473`, `0.759`, and `0.895`. Because `0.895` is at or above `0.8`, the
fourth token is removed.

In stage 4, after renormalizing over the three remaining tokens, token 0 has ratio
$0.151 / 0.528 = 0.29$ of the top probability, which is below `0.3`, so it is removed.
In stage 5, dividing by `2.0` flattens the two survivors, producing cumulative sums
`[0, 0.562, 1.0, ...]`. In stage 6, `u = 0.3` selects token `1` and `u = 0.7` selects
token `2`.

Dividing by temperature first changes the result. At temperature `2.0`, the penalized
distribution is `[0.168, 0.314, 0.245, 0.149, 0.090, 0.034]`. Because this distribution
is flatter, the same `top_p` and `min_p` thresholds keep tokens `0`, `1`, and `2`, and
token `0` is sampled 23% of the time instead of never.

## Constraints

- $1 \le B \le 4096$, $2 \le V \le 1000$, and $1 \le T \le 64$. Finite logits lie in
  $[-100, 100]$.
- `chain_logits` is compared against the reference with a relative tolerance of `1e-4`
  and an absolute tolerance of `1e-6`. Entries equal to `-inf` must match exactly.
  Sampled tokens are compared exactly.
- Tests avoid borderline comparisons. Logits at the top-k boundary differ by at least
  `1e-3`, every partial probability sum differs from `top_p` by at least `1e-3`, every
  probability ratio differs from `min_p` by at least `0.1%`, and $u \cdot c_{V-1}$ is at
  least `1e-4` away from every cumulative sum. Ties between finite logits do not occur.
- **Statistical cases.** `sample_chain` is called once with 4000 identical rows and 4000
  independent uniform values. The empirical token distribution is compared against the
  exact distribution of the chain using a finite-sample Chernoff bound. A correct
  solution fails with probability below $10^{-9}$, and because the seeds are fixed, it
  either always passes or always fails.
- You do not need Python loops over rows or tokens.

## Why this matters

Every request to an LLM server carries these sampling parameters, and every engine
applies them in a fixed order that the API does not expose. Engines disagree on that
order. llama.cpp applies temperature last, while Hugging Face transformers and vLLM
apply it first. As a result, `temperature=1.5, top_p=0.9` selects a different set of
tokens and produces noticeably different text depending on which server sits behind the
same OpenAI-compatible endpoint.

With temperature last, top-p and min-p trim the tail of the model's unscaled
distribution, so a high temperature only reshuffles already plausible tokens. With
temperature first, a high temperature widens the nucleus and lets the tail back in,
turning creative settings into gibberish. Getting the stage order wrong raises no error
because the sampler still returns valid tokens, just from a distribution nobody asked
for.
