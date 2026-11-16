Compose the samplers of the earlier problems into the chain that turns a row of
logits into a token: penalties, top-k, top-p, min-p, temperature, then one random
draw. Each stage is a few lines, and the kit supplies the three filters. The exercise
is the **order**: every stage works on what the stage before it left, so a different
order keeps a different set of tokens.

## Specification

```python
def chain_logits(logits, cfg, history): ...     # the logits after stages 1 to 5
def sample_chain(logits, cfg, history, u): ...  # one token per row
```

`cfg` is a dict with exactly these keys:

| key | type | range | value that switches the stage off |
|---|---|---|---|
| `"repetition_penalty"` | float | $\ge 1$ | `1.0` |
| `"frequency_penalty"` | float | $\ge 0$ | `0.0` |
| `"presence_penalty"` | float | $\ge 0$ | `0.0` |
| `"top_k"` | int | $\ge 0$ | `0`, or any value $\ge V$ |
| `"top_p"` | float | $0 < p \le 1$ | `1.0` |
| `"min_p"` | float | $0 \le m < 1$ | `0.0` |
| `"temperature"` | float | $> 0$ | `1.0` |

The stages run in the default order of llama.cpp. Each row is processed
independently; $x$ is one row of logits, and "removing" a token means setting its
logit to `-inf`.

**1. Penalties.** Let $c_v$ be the number of times token $v$ occurs in the row's
`history`. For every token with $c_v > 0$, first the repetition penalty $\theta$,
then the frequency penalty $\alpha_f$ and the presence penalty $\alpha_p$:

$$
x_v \leftarrow
\begin{cases}
x_v / \theta & \text{if } x_v > 0, \\
x_v \cdot \theta & \text{if } x_v \le 0,
\end{cases}
\qquad\text{then}\qquad
x_v \leftarrow x_v - c_v\,\alpha_f - \alpha_p .
$$

Tokens that do not occur in the history are unchanged. The repetition and presence
penalties are charged once per token, the frequency penalty once per occurrence.

**2. Top-k.** Keep the `top_k` largest logits and remove the rest.

**3. Top-p.** Take the softmax of the current logits (removed tokens have probability
0, so this is the distribution **renormalized over the survivors of top-k**). Rank
the tokens by decreasing probability and keep a token exactly when the probability
mass ranked strictly before it is below `top_p`. The token that carries the mass to
or past `top_p` is kept.

**4. Min-p.** Take the softmax of the current logits again and remove every token
whose probability is below `min_p` times the largest probability.

**5. Temperature.** Divide every logit by `temperature`. Removed tokens stay `-inf`.

`chain_logits` returns the result of stage 5.

**6. Sample.** `sample_chain` draws one token per row from
$\pi = \operatorname{softmax}(\texttt{chain\_logits}(\ldots))$ by inverse-CDF
sampling with the row's own uniform number: with $c_k = \sum_{i \le k} \pi_i$, the
token is the smallest $k$ with $c_k > u \cdot c_{V-1}$. With `u` uniform on
$[0, 1)$, token $k$ is drawn with probability $\pi_k$; a removed token is never
drawn.

A stage that is switched off leaves the logits exactly as they are. In particular
`top_p = 1.0` means there is no top-p stage: nothing may be removed, however small
its probability.

The filters of stages 2 to 4 are in the kit, and you may use them:

```python
from learnai.kit import ops

ops.top_k_filter(logits, k)       # k <= 0 or k >= V: unchanged
ops.top_p_filter(logits, p)
ops.min_p_filter(logits, min_p)   # min_p = 0: unchanged
ops.softmax(logits)               # -inf entries get probability 0
```

Each works on the last axis and returns logits: removed entries are `-inf`, kept
entries are unchanged, and entries that were already `-inf` stay `-inf`. Equal
logits are ranked by token index, lowest first. Any implementation of the six stages
that follows the definitions above is accepted.

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (already
  masked, for example by a grammar); every row has at least one finite entry.
- `history`: an integer tensor of shape `(B, T)`, $T \ge 1$, with values in
  $[0, V)$: the tokens each row has produced so far. All of them count.
- `u`: a float32 tensor of shape `(B,)` with $0 \le u < 1$.
- `chain_logits` returns a float32 tensor of shape `(B, V)`; `sample_chain` returns
  an integer tensor of shape `(B,)`.

Randomness is an input: do not draw random numbers, and do not call your framework's
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

Penalties: token 0 occurred twice and has a positive logit,
$3.0 / 1.2 - 2 \cdot 0.5 - 0.25 = 1.25$; token 5 occurred once and has a negative
one, $-1.0 \cdot 1.2 - 0.5 - 0.25 = -1.95$. Token 0 was the most likely token and is
now third. Top-p: ranked, the survivors of top-k have probabilities 0.473, 0.287,
0.135, 0.105, and the mass before them is 0, 0.473, 0.759, 0.895; the fourth is at
or above 0.8 and goes. Min-p: token 0 now has $0.151 / 0.528 = 0.29$ of the top
probability, which is below 0.3. The temperature then flattens what is left, and the
cumulative sums `[0, 0.562, 1.0, ...]` send `u = 0.3` to token 1 and `u = 0.7` to
token 2.

Dividing by the temperature first gives a different answer. At temperature 2 the
penalized distribution is `[0.168, 0.314, 0.245, 0.149, 0.090, 0.034]`: flatter, so
the same top-p and min-p keep tokens 0, 1, and 2, and token 0 is sampled 23% of the
time instead of never.

## Constraints

- $1 \le B \le 4096$, $2 \le V \le 1000$, $1 \le T \le 64$. Finite logits lie in
  $[-100, 100]$.
- `chain_logits` is compared with the reference to a relative tolerance of `1e-4`
  and an absolute tolerance of `1e-6`; `-inf` entries must match exactly. Tokens are
  compared exactly.
- No comparison is close in the tests: the logits at the top-k boundary differ by at
  least `1e-3`, every partial sum of probabilities differs from `top_p` by at least
  `1e-3`, every probability ratio differs from `min_p` by at least 0.1 percent, and
  $u \cdot c_{V-1}$ is at least `1e-4` from every cumulative sum. Ties between finite
  logits do not occur.
- **Statistical cases.** `sample_chain` is called once with 4000 identical rows and
  4000 independent uniforms, and the empirical distribution of the tokens is
  compared with the exact distribution of the chain using a finite-sample (Chernoff)
  bound. A correct solution fails with probability below $10^{-9}$, and since the
  seeds are fixed it either always passes or always fails.
- No Python loop over rows or tokens is needed.

## Why this matters

Every request to an LLM server carries these parameters, and every engine applies
them in a fixed order that the API does not show. The orders differ: llama.cpp
applies the temperature last, Hugging Face transformers and vLLM apply it first. So
`temperature=1.5, top_p=0.9` selects a different set of tokens, and produces
visibly different text, depending on which server is behind the same
OpenAI-compatible endpoint. With the temperature last, top-p and min-p cut the
tail of the model's own distribution and a high temperature can only reshuffle
probable tokens; with the temperature first, a high temperature widens the nucleus
and lets the tail back in, which is how "creative" settings turn into gibberish.
Getting the chain wrong produces no error: the sampler still returns valid tokens,
just from a distribution nobody asked for.
