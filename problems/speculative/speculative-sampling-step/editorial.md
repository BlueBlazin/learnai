## The idea

We want a sample from the target distribution `p`, but the sample we were handed came
from the draft distribution `q`. Speculative sampling is a rejection scheme with a
correction step that makes the result exact with a single proposal.

**Acceptance.** Keep the draft token `x` with probability $\min(1, p[x]/q[x])$. Where
the draft model underrates a token ($q \le p$) its proposals are always kept; where it
overrates one, proposals are thinned by exactly the factor needed. The probability
that token $t$ is drafted *and* accepted is

$$
q[t] \cdot \min\!\Big(1, \frac{p[t]}{q[t]}\Big) = \min(p[t], q[t]).
$$

**Residual.** That is at most $p[t]$, and the shortfall is
$p[t] - \min(p[t], q[t]) = \max(0, p[t] - q[t])$. The total probability of a rejection
is

$$
1 - \beta = 1 - \sum_t \min(p[t], q[t]) = \sum_t \max(0, p[t] - q[t]) = R,
$$

the same number as the total shortfall. So if a rejected step emits a token from the
normalized residual $\max(0, p - q) / R$, token $t$ receives $R \cdot \max(0, p[t] -
q[t]) / R$ through that route, and

$$
P(\text{emit } t) = \min(p[t], q[t]) + \max(0, p[t] - q[t]) = p[t].
$$

This is the whole proof of losslessness, and it shows what each ingredient is for:
the ratio makes the accepted mass $\min(p, q)$, the positive part of $p - q$ is what
is missing, and the normalization constant is the rejection probability itself.

## Reference solution (PyTorch)

```python
def speculative_step(p, q, x, u_accept, u_resample):
    if bool(u_accept * q[x] < p[x]):
        return x, True
    cdf = torch.cumsum(torch.clamp(p - q, min=0.0), dim=0)
    token = int((cdf <= u_resample * cdf[-1]).sum())
    return token, False
```

Three details make it robust:

- `u_accept * q[x] < p[x]` is the acceptance test without a division or a `min`.
  For $p[x] \ge q[x]$ it holds for every $u < 1$; for $p[x] = 0$ it fails for every
  $u \ge 0$. The strict inequality gives probability exactly $p[x]/q[x]$ for $u$
  uniform on $[0, 1)$.
- The residual is never divided by its total. Comparing the cumulative sum with
  `u_resample * cdf[-1]` is the same test, and since the threshold is strictly below
  the last cumulative sum, the count can never run past the last token with positive
  residual.
- Counting the entries `<=` the threshold returns the first index that exceeds it. A
  token with residual 0 repeats the previous cumulative sum, so it is either counted
  together with its predecessor or skipped together with it; it is never selected.

## Common mistakes

- **Always accepting**, or accepting whenever the target's argmax agrees. The output
  is then distributed as `q`. Text quality drops to the draft model's while every
  speed metric improves.
- **Inverting the ratio** to $q[x]/p[x]$. It is easy to do, because the two papers use
  opposite letters for the two models. A sanity check: a token the target gives
  probability 0 must never be accepted.
- **Accepting only when $p[x] \ge q[x]$.** That discards the partial credit: a token
  with $p[x] < q[x]$ must still be accepted with probability $p[x]/q[x]$.
- **Resampling from `p`.** It looks natural ("fall back to the target model"), but
  the accepted mass $\min(p, q)$ was already delivered, so tokens on which the models
  agree are overcounted and the others undercounted.
- **Using $|p - q|$** as the residual. Half of that mass sits on tokens the draft
  model overrates, which are exactly the ones that must not receive more.
- **Not normalizing the residual.** Its total is $R = 1 - \beta < 1$. Comparing its
  cumulative sum with a raw uniform sends every $u \ge R$ past the end (an index
  error, or after clamping, a spike on the last token).
- **Reusing the uniform.** Given a rejection, `u_accept` is no longer uniform on
  $[0, 1)$: it is known to be at least $p[x]/q[x]$. Using it again for the inverse
  CDF biases the resampled token toward high indices. Each decision needs its own
  independent random number.

## Complexity

$O(1)$ for an acceptance and $O(V)$ for a rejection (one subtraction, one cumulative
sum, one comparison), with $O(V)$ memory.

## In the wild

The scheme was published independently by Leviathan, Kalman, and Matias, "Fast
Inference from Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192),
whose notation this problem follows (`p` target, `q` draft; resample from
$\operatorname{norm}(\max(0, p - q))$), and by Chen, Borgeaud, Irving, Lespiau, Sifre,
and Jumper, "Accelerating Large Language Model Decoding with Speculative Sampling"
(2023, arXiv:2302.01318), who call it modified rejection sampling and write `q` for
the target and `p` for the draft, with the residual $(q - p)_+$. Both prove that the
output follows the target distribution, and Leviathan et al. show that the acceptance
probability is $\beta = \sum_t \min(p_t, q_t)$, one minus the total variation
distance between the two models, which is what determines the speedup.

A full decoder applies this step to a block of $\gamma$ draft tokens. The draft model
generates them autoregressively; the target model scores all $\gamma + 1$ positions
in one forward pass; the tokens are verified left to right; at the first rejection
the replacement is emitted and the rest of the block is discarded; if every token is
accepted, one more token is sampled from the target's distribution at the last
position, so each target pass yields between 1 and $\gamma + 1$ tokens. Hugging Face
`transformers` (assisted generation) and vLLM implement this verification batched
over sequences and positions, with `torch.rand` supplying the uniforms. With greedy
decoding the rule degenerates to "accept if the draft token is the target's argmax".
