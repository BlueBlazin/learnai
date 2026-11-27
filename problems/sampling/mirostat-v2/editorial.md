## The idea

The quality of sampled text tracks its **surprise**, defined as the average of
$-\log_2 p(\text{token})$ over the generated tokens (whose base-2 exponential is the
perplexity). If surprise is too low, the text falls into repetitive loops ("the
boredom trap"). If surprise is too high, the text drifts into incoherence ("the
confusion trap"). Top-k and top-p control surprise only indirectly, so the actual
perplexity shifts with the peakedness of each step's distribution and with sequence
length.

Mirostat (Basu et al., 2021) turns decoding into a feedback control loop around a
target surprise $\tau$:

- **Actuator.** The truncation threshold $\mu$ removes all tokens with surprise
  above $\mu$. Lowering $\mu$ restricts sampling to more probable tokens and reduces
  the expected surprise of the next draw.
- **Measurement.** The observed surprise $s$ of the token that was actually drawn.
- **Controller.** A single integral feedback update,
  $\mu \leftarrow \mu - \eta\,(s - \tau)$.

Mirostat v1 converted $\mu$ into a dynamic top-$k$ cutoff by estimating the Zipf
exponent of the distribution at every step. Version 2 removes that estimation step
and truncates directly by surprise, which is what this problem implements.

**Why the mean converges to the target.** Summing the update rule over $N$ steps
gives

$$
\mu_N = \mu_0 - \eta \sum_{t < N} (s_t - \tau)
\quad\Longrightarrow\quad
\frac{1}{N} \sum_{t<N} s_t \;=\; \tau + \frac{\mu_0 - \mu_N}{\eta N}.
$$

This is an exact algebraic identity, not an approximation. As long as $\mu$ stays
bounded, the mean surprise differs from $\tau$ by $O(1/N)$. In practice $\mu$ stays
bounded because high $\mu$ admits surprising tokens (which push $\mu$ down) and low
$\mu$ leaves only likely tokens (which push $\mu$ up). The long-run tests measure
$s$ directly from the returned tokens in bits under the model's original
distribution. If an implementation feeds a different quantity into the update, the
identity holds for *that* quantity instead, and the true mean surprise converges to
the wrong value.

## Reference solution (PyTorch)

```python
def mirostat_step(logits, mu, tau, eta, u):
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    surprise = -log_p / math.log(2.0)
    top = torch.argmax(logits, dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    keep = (surprise <= mu[:, None]) | (index == top)
    cdf = torch.cumsum(torch.where(keep, torch.exp(log_p), 0.0), dim=-1)
    token = (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
    observed = torch.take_along_dim(surprise, token[:, None], dim=-1)[:, 0]
    return token, mu - eta * (observed - tau)
```

- Computing surprise from log-softmax avoids underflow. Taking `log2(softmax(...))`
  turns small positive probabilities into `0.0` and then into infinite surprise, even
  for unmasked tokens.
- No sort is needed. Candidate-list implementations sort tokens and cut at the first
  token above $\mu$, whereas on a dense tensor the same kept set is a single
  elementwise comparison.
- Always keeping `argmax(logits)` handles the empty-set fallback without branching.
  Whenever any token satisfies $s_i \le \mu$, the most likely token already satisfies
  it too because it has the smallest surprise.
- `cdf[:, -1:]` is the total kept mass $c_{V-1}$. Scaling `u` by `cdf[:, -1:]`
  samples from the renormalized distribution without dividing.

## Common mistakes

- **Using natural logarithms.** `-log_p` measures surprise in nats rather than bits.
  The controller still converges, but to $\tau$ nats, which equals
  $\tau / \ln 2 \approx 1.44\,\tau$ bits. At $\tau = 3$, the generated text ends up
  at 4.3 bits (perplexity 20 instead of 8). Because recommended $\tau$ values are
  published in bits, this causes a silent shift in output quality.
- **Wrong update sign.** Using $\mu \leftarrow \mu + \eta\,(s - \tau)$ creates
  positive feedback. A surprising token raises $\mu$, which admits even more
  surprising tokens, until $\mu$ drifts off and the sampler degenerates into either
  untruncated sampling or greedy decoding.
- **Using surprise after renormalization.** Renormalizing the kept probabilities
  divides each kept $p_i$ by the kept mass, making the drawn token look more likely
  and less surprising. Feeding $-\log_2$ of that renormalized probability into the
  update holds the *truncated* surprise at $\tau$, pushing the true model surprise
  higher (around 3.7 bits instead of 3 on the long-run test). See "In the wild" for
  why this variant is common.
- **Missing empty-set fallback.** $\mu$ can drop below every token's surprise and can
  even become negative. Without keeping the most likely token, the kept mass becomes
  zero and the inverse-CDF scan returns a degenerate index, typically the last token
  in the vocabulary.
- **Truncating at $\tau$ instead of $\mu$.** This updates $\mu$ without ever using
  it, turning the sampler into a static surprise cutoff whose mean surprise sits well
  below $\tau$.
- **Skipping truncation or forgetting to scale `u` by the kept mass.** Skipping
  truncation samples from the full distribution. Forgetting to multiply `u` by
  `cdf[:, -1:]` compares `u` against unnormalized cumulative sums, so any `u` above
  the kept mass falls off the end of the vocabulary.
- **Sampling in sorted order.** Porting a candidate-list implementation often walks
  tokens from most to least likely. That defines the same distribution, but this
  problem fixes token order so that a given `u` deterministically selects a specific
  token.
- **Sharing one $\mu$ across the batch.** Each sequence has its own feedback state.
  Averaging $\mu$ across rows or storing a single scalar on a shared sampler couples
  unrelated requests.
- **Drawing your own randomness.** Calling `torch.multinomial` or similar RNG
  functions breaks deterministic testing and is forbidden here.

## Complexity

$O(BV)$ time and memory per step, consisting of a log-softmax, a comparison, and a
cumulative sum. Unlike top-p, Mirostat v2 requires no sort. The per-sequence state is
a single float.

## In the wild

- **The paper.** Basu, Ramachandran, Keskar, and Varshney, "Mirostat: A Neural Text
  Decoding Algorithm that Directly Controls Perplexity" (ICLR 2021). The authors'
  code (`basusourya/mirostat`, `mirostat.py`) computes the observed surprise as
  `math.log2(1 / prob_original[prev_i])`, using the pre-truncation probability. This
  problem follows that definition.
- **llama.cpp and text-generation-webui.** In `llama_sampler_mirostat_v2_apply`
  (`src/llama-sampler.cpp`), llama.cpp computes softmax, cuts the sorted candidates
  at the first token with `-log2f(p) > mu` (keeping at least one), re-runs softmax,
  samples, and updates with `observed_surprise = -log2f(cur_p->data[idx].p)` using
  the **renormalized** probability. It initializes and resets `mu` to `2.0f * tau`.
  text-generation-webui's `MirostatLogitsWarper` does the same with
  `-math.log2(prob_topk[prev_i])`. Both engines therefore hold the renormalized
  surprise at $\tau$, which leaves the model's true surprise higher by the average
  $-\log_2$ of the kept mass. This is one reason perplexities differ across inference
  engines at the same `tau`.
- **Defaults.** llama.cpp defaults to `tau = 5.0` and `eta = 0.1`. Mirostat replaces
  truncation samplers such as top-k, top-p, and min-p, and sits at the end of the
  sampler chain because it draws the token itself.
- **State.** Production servers must reset `mu` per sequence, save it with session
  state, and store it as a per-row vector during batched decoding.
