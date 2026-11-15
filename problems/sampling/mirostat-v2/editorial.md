## The idea

The quality of sampled text tracks its **surprise**: the average of
$-\log_2 p(\text{token})$ over the generated tokens, whose exponential is the
perplexity. Too little and the text loops ("the boredom trap"); too much and it
drifts into nonsense ("the confusion trap"). Top-k and top-p control surprise only
indirectly, and the surprise they produce changes with the shape of each step's
distribution and with the length of the text.

Mirostat (Basu et al., 2021) closes the loop. It picks a target $\tau$, and treats
generation as a control problem:

- the **actuator** is a truncation threshold $\mu$: tokens with surprise above $\mu$
  cannot be sampled, so lowering $\mu$ lowers the expected surprise of the next
  token;
- the **measurement** is the surprise $s$ of the token that was actually drawn;
- the **controller** is one line of integral feedback,
  $\mu \leftarrow \mu - \eta\,(s - \tau)$.

Version 1 of the algorithm turns $\mu$ into a top-k cutoff through an estimate of
the Zipf exponent of the distribution. Version 2 drops the estimate and truncates by
surprise directly, which is this problem.

**Why the mean lands on the target.** Sum the update over $N$ steps:

$$
\mu_N = \mu_0 - \eta \sum_{t < N} (s_t - \tau)
\quad\Longrightarrow\quad
\frac{1}{N} \sum_{t<N} s_t \;=\; \tau + \frac{\mu_0 - \mu_N}{\eta N}.
$$

This is an identity, not an approximation. As long as $\mu$ stays bounded, and it
does (when $\mu$ is high the kept set is large, tokens are surprising, and $\mu$
falls; when it is low only likely tokens remain and it rises), the mean surprise
differs from $\tau$ by $O(1/N)$. The long-run tests check exactly this, and they
measure $s$ themselves, from the returned tokens, in bits under the model's
distribution. A solution that feeds a different number into the update still
satisfies the identity for *its* number, and the mean of the real surprise ends up
somewhere else.

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

- The surprise comes from a log-softmax, not from `log2(softmax(...))`: a
  probability that underflows to 0 would become an infinite surprise although the
  token is not masked.
- There is no sort. Implementations that work on a sorted candidate list cut it at
  the first token above $\mu$; on a dense tensor the same set is one comparison,
  because surprise is monotone in probability.
- "Keep the argmax always" replaces the special case "if nothing is kept, keep the
  best token": whenever any token passes, the least surprising one passes too.
- `cdf[:, -1:]` is the kept mass. Multiplying `u` by it samples from the
  renormalized distribution without dividing.

## Common mistakes

- **Natural logarithms.** `-log_p` is surprise in nats. The loop still converges,
  to $\tau$ nats, which is $\tau / \ln 2 = 1.44\,\tau$ bits: with $\tau = 3$ the
  text has a surprise of 4.3 bits, a perplexity of 20 instead of 8. Since $\tau$ is
  a user-facing parameter with published recommended values in bits, this is a
  silent quality change.
- **The wrong sign.** $\mu \leftarrow \mu + \eta\,(s - \tau)$ is positive feedback:
  a surprising token raises the limit, which admits more surprising tokens. $\mu$
  runs off in one direction and the sampler degenerates into plain sampling or
  into greedy decoding.
- **Surprise after renormalization.** After truncation the kept probabilities are
  renormalized, and the drawn token's renormalized probability is larger than its
  probability under the model, by the factor 1 / (kept mass). Feeding
  $-\log_2$ of that into the update makes the controller hold the *truncated*
  distribution's surprise at $\tau$, and the true surprise sits above it: 3.7
  instead of 3 bits on the long-run test. See "In the wild": this variant is
  widespread.
- **No fallback for an empty set.** $\mu$ can fall below the surprise of every
  token (it can even go negative). Without the rule that the most likely token
  stays, the kept mass is 0 and the sampler returns whatever index a degenerate
  cumulative sum produces, typically the last token of the vocabulary.
- **Truncating at $\tau$** instead of $\mu$. Then $\mu$ is updated and never
  used, and the sampler is a fixed surprise cutoff whose mean lies well below the
  target.
- **Not truncating at all**, or **forgetting to scale $u$ by the kept mass**. The
  first samples from the full distribution; the second compares $u$ with
  unnormalized cumulative sums, so every $u$ above the kept mass falls off the end.
- **Sampling in sorted order.** A port of a candidate-list implementation walks the
  tokens from most to least likely. The distribution is the same, but this
  problem fixes token order so that a given `u` selects a given token.
- **One $\mu$ for the batch.** The limit is per-sequence state. Averaging it (or
  keeping a scalar on a sampler object shared by a batch) couples unrelated
  requests.
- **Drawing your own randomness** with `torch.multinomial`: not reproducible, and
  ruled out here.

## Complexity

$O(BV)$ time and memory per step: a log-softmax, a comparison, and a cumulative
sum. No sort, unlike top-p. The state is one float per sequence.

## In the wild

- **The paper.** Basu, Ramachandran, Keskar, and Varshney, "Mirostat: A Neural Text
  Decoding Algorithm that Directly Controls Perplexity" (ICLR 2021). The authors'
  code (`basusourya/mirostat`, `mirostat.py`) computes the observed surprise as
  `math.log2(1 / prob_original[prev_i])`: the probability from before the
  truncation. This problem follows that.
- **llama.cpp** (`llama_sampler_mirostat_v2_apply` in `src/llama-sampler.cpp`):
  softmax, cut the sorted candidates at the first one with `-log2f(p) > mu`, keep
  at least one, softmax again, sample, and then
  `observed_surprise = -log2f(cur_p->data[idx].p)` with the **renormalized** `p`.
  `mu` starts at `2.0f * tau` and is reset to it with the sampler.
  text-generation-webui's `MirostatLogitsWarper` does the same
  (`-math.log2(prob_topk[prev_i])`). So the two most used implementations hold the
  renormalized surprise at $\tau$; the surprise of the text under the model is
  higher, by $-\log_2$ of the kept mass on average. If you compare perplexities
  across engines at "the same `tau`", this is one reason they differ.
- **Defaults.** llama.cpp uses `tau = 5.0` and `eta = 0.1`. Mirostat replaces the
  truncation samplers (top-k, top-p, min-p are not applied with it), and because it
  draws the token itself it sits at the end of the chain.
- **State.** `mu` must be reset per sequence and saved with a session. In a batched
  server it is a vector, as here.
