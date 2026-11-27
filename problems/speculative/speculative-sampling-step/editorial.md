## The idea

We want a token sampled from the target distribution `p`, but we are given a proposal
`x` sampled from the draft distribution `q`. Speculative sampling uses rejection
sampling with a residual correction so that a single proposal produces an exact sample
from `p`.

**Acceptance.** Keep the draft token `x` with probability $\min(1, p[x] / q[x])$.
When the draft model underrates a token ($q[t] \le p[t]$), its proposals are always
accepted. When it overrates a token ($q[t] > p[t]$), its proposals are thinned by the
exact factor needed. The probability that token $t$ is both drafted and accepted is

$$
q[t] \cdot \min\!\Big(1, \frac{p[t]}{q[t]}\Big) = \min(p[t], q[t]).
$$

Let $\beta = \sum_t \min(p[t], q[t])$ be the overall acceptance probability.

**Residual.** The accepted mass $\min(p[t], q[t])$ is at most $p[t]$, leaving a
shortfall of $p[t] - \min(p[t], q[t]) = \max(0, p[t] - q[t])$ for each token $t$.
Because both `p` and `q` sum to 1, the total rejection probability equals the total
shortfall $R$:

$$
1 - \beta = 1 - \sum_t \min(p[t], q[t]) = \sum_t \max(0, p[t] - q[t]) = R.
$$

If every rejection samples a replacement from the normalized residual
$\max(0, p - q) / R$, token $t$ receives $R \cdot \max(0, p[t] - q[t]) / R$ from the
resampling path. Adding both paths gives

$$
P(\text{emit } t) = \min(p[t], q[t]) + \max(0, p[t] - q[t]) = p[t].
$$

That completes the proof of losslessness. The acceptance ratio delivers
$\min(p, q)$, the positive part of $p - q$ supplies the missing mass, and the
normalization constant $R$ equals the rejection probability itself.

## Reference solution (PyTorch)

```python
def speculative_step(p, q, x, u_accept, u_resample):
    if bool(u_accept * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
```

Three implementation details make this short and numerically safe:

- Writing `u_accept * q[x] < p[x]` avoids division and `min`. When $p[x] \ge q[x]$,
  the inequality holds for every $u_\text{accept} \in [0, 1)$. When $p[x] = 0$, it
  fails for every $u_\text{accept} \ge 0$. The strict inequality gives acceptance
  probability exactly $p[x] / q[x]$.
- Scaling the uniform threshold by `cdf[-1]` avoids dividing the residual by $R$.
  Because `u_resample < 1`, the threshold `u_resample * cdf[-1]` is strictly below
  the final cumulative sum, so the index never runs past the last token with positive
  residual.
- Counting entries where `cdf <= threshold` returns the first index where `cdf`
  exceeds the threshold. Any token with zero residual repeats the previous cumulative
  sum, so it is either counted or skipped along with its predecessor and is never
  selected.

## Common mistakes

- **Always accepting**, or accepting whenever the target's argmax agrees. The output
  then follows `q` instead of `p`. Quality drops to the draft model's level while
  every speed metric looks better.
- **Inverting the ratio** to $q[x] / p[x]$. This is easy to do because the two
  foundational papers swap the letters `p` and `q`. As a sanity check, a token with
  $p[x] = 0$ must never be accepted.
- **Accepting only when $p[x] \ge q[x]$.** This throws away partial credit. A token
  with $p[x] < q[x]$ still needs to be accepted with probability $p[x] / q[x]$.
- **Resampling from `p`.** Falling back directly to `p` looks natural, but the
  accepted mass $\min(p, q)$ was already emitted. Tokens where both models agree get
  overcounted, and the rest get undercounted.
- **Using $|p - q|$ as the residual.** Half of that mass sits on tokens the draft
  model overrated, which are the exact tokens that must not receive extra probability.
- **Not normalizing the residual.** The unnormalized residual sums to
  $R = 1 - \beta < 1$. Comparing its cumulative sum against an unscaled uniform sends
  every $u_\text{resample} \ge R$ past the end of the array, causing an out-of-bounds
  index or a probability spike on the last token after clamping.
- **Reusing `u_accept` for resampling.** Conditioned on rejection, `u_accept` is no
  longer uniform on $[0, 1)$ because it is at least $p[x] / q[x]$. Reusing it for the
  inverse CDF biases resampled tokens toward higher indices. Each decision needs its
  own independent uniform draw.

## Complexity

$O(1)$ time on acceptance and $O(V)$ time on rejection (one subtraction, one
cumulative sum, and one comparison), with $O(V)$ memory, where $V$ is the vocabulary
size.

## In the wild

The algorithm was published independently by two groups. Leviathan, Kalman, and
Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023,
arXiv:2211.17192), use the notation followed here (`p` for target, `q` for draft, and
residual $\operatorname{norm}(\max(0, p - q))$). Chen, Borgeaud, Irving, Lespiau,
Sifre, and Jumper, "Accelerating Large Language Model Decoding with Speculative
Sampling" (2023, arXiv:2302.01318), call it modified rejection sampling and swap the
letters (`q` for target, `p` for draft, and residual $(q - p)_+$). Both prove
exactness, and Leviathan et al. show that the acceptance rate
$\beta = \sum_t \min(p_t, q_t)$ equals one minus the total variation distance between
`p` and `q`, which governs the expected speedup.

A full speculative decoder applies this step to a block of $\gamma$ draft tokens.
First, the draft model generates $\gamma$ tokens autoregressively, and the target
model scores all $\gamma + 1$ positions in a single forward pass. The decoder then
verifies the draft tokens from left to right. At the first rejection, it emits the
resampled replacement and discards the rest of the block. If all $\gamma$ tokens are
accepted, it samples one bonus token from the target's distribution at the final
position, so each target forward pass yields between 1 and $\gamma + 1$ tokens.
Hugging Face `transformers` (assisted generation) and vLLM implement this
verification batched across sequences and positions using `torch.rand` for the
uniforms. Under greedy decoding, the rule simplifies to accepting a draft token if it
matches the target's argmax.
