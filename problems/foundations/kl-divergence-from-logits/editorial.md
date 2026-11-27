## The idea

$\operatorname{KL}(P \,\|\, Q) = \sum_i P_i (\log P_i - \log Q_i)$ is the expected number of extra nats you pay when encoding samples from $P$ with a code built for $Q$. In model evaluation, $P$ is the reference model you trust (such as full precision or a target model) and $Q$ is the model you are testing (quantized, half precision, or a draft). The order matters because the sum is weighted by $P$, so it measures error on the tokens the reference model actually predicts.

Three things go wrong when computing this formula in floating point.

**Probabilities underflow, while log-probabilities do not.** A float32 probability below about $10^{-45}$ rounds to `0.0`, and its logarithm becomes `-inf`. By contrast, the log-probability from `log_softmax` stays a finite negative number. Both logarithms must therefore come from `log_softmax`, never from `log(softmax(...))`. The leading factor $P_i$ may underflow freely, because `0.0` times a finite log-ratio is `0.0`, which is the correctly rounded value of a term smaller than $10^{-40}$.

**Zero probabilities need explicit rules.** Masked tokens produce $0 \cdot (-\infty)$ and $-\infty - (-\infty)$, both of which evaluate to NaN. The mathematical convention is fixed:

- $0 \log(0 / q) = 0$, because $x \log x \to 0$ as $x \to 0$. A token that $P$ never produces costs nothing, regardless of $Q$.
- $p \log(p / 0) = +\infty$ for $p > 0$. When $Q$ assigns zero probability to a token that $P$ can produce, no finite penalty suffices.

The subtle part is *what counts as zero*. A `-inf` logit is an exact zero, whereas a finite logit represents a strictly positive probability even at $e^{-200}$. Checking `p == 0` after exponentiating confuses underflow with masking and returns a finite divergence when the true answer is `+inf`. Always decide zero probabilities from the input logits.

**Rounding can cross zero.** By Gibbs' inequality, the exact sum is $\ge 0$, with equality only when $P = Q$. Individual terms have both positive and negative signs, so for nearly identical distributions they cancel down to float32 rounding noise, which lands below zero about half the time. Clamping at `0.0` prevents a downstream `sqrt`, log-scale plot, or `>= 0` assertion from failing.

## Reference solution (PyTorch)

```python
def kl_from_logits(p_logits, q_logits):
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p_zero = torch.isneginf(p_logits)
    q_zero = torch.isneginf(q_logits)
    terms = torch.exp(log_p) * (log_p - log_q)
    terms = torch.where(q_zero, torch.inf, terms)  # rule 2
    terms = torch.where(p_zero, 0.0, terms)  # rule 1 wins where both are masked
    return terms.sum(dim=-1).clamp(min=0.0)
```

The unconditional product produces NaN only at entries that one of the two `where` calls overwrites, so no NaN reaches the sum. Applying the `p_zero` mask second ensures that rule 1 takes precedence over rule 2.

The JAX version is the same using `jax.nn.log_softmax` and `jnp.where`. MLX does not expose `log_softmax` in `mlx.core`, and writing `x - mx.logsumexp(x, axis=-1, keepdims=True)` loses accuracy at large magnitudes. For a row of logits near $10^4$, the log-sum-exp is also near $10^4$ (where float32 spacing is about $10^{-3}$), so subtracting it inherits that rounding error. The MLX reference shifts by the row maximum first, just like `stable-softmax`:

```python
shifted = x - mx.max(x, axis=-1, keepdims=True)
log_p = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
```

## Common mistakes

- **Swapping the arguments.** `torch.nn.functional.kl_div(input, target)` takes the *approximation* first (as log-probabilities) and the reference second, the reverse of mathematical notation. Many swapped-KL bugs in evaluation code come from that habit.
- **No rule for zeros.** A single masked token in $P$ turns the row into NaN, which turns a dataset mean into NaN.
- **Working on probabilities.** `log(softmax(q))` produces `-inf` for a confident model's unlikely tokens, turning a finite divergence such as 99.3 nats into `+inf`.
- **Testing $P_i = 0$ on the underflowed probability.** This handles masked tokens but fails when $P$ has a small finite logit that $Q$ masks, where the divergence must be `+inf`.
- **Skipping tokens masked in $Q$.** Ignoring `-inf` in `q_logits` looks like defensive masking, but it silently reports a low divergence for a model that forbids a token $P$ predicts.
- **Using `x - logsumexp(x)` without shifting first.** This works for logits near 1, but loses up to $10^{-3}$ nats per row for logits near $10^4$—an error larger than the divergence between a model and its 8-bit quantization.
- **Returning the unclamped sum.** Rounding error produces slightly negative values for nearly identical distributions.
- **Reducing over the batch.** Following `reduction="batchmean"` returns a single scalar. Here the caller wants one value per row so it can average over non-padding positions itself.

## Complexity

Computing `log_softmax` twice and combining the terms elementwise takes $O(V)$ time and memory per row.

## In the wild

`llama.cpp`'s `perplexity` tool includes a `--kl-divergence` mode. You save the base model's logits with `--kl-divergence-base`, then compare a quantized model against them with mean KL divergence per token as the headline metric. In speculative decoding, the acceptance probability of a draft token is $\min(1, P_i / Q_i)$ (Leviathan et al., 2023), and the total variation distance governing acceptance rate is bounded by $\sqrt{\mathrm{KL}/2}$ by Pinsker's inequality. That bound is why draft models are screened by KL divergence first.

Float32 summation still carries rounding noise. For two nearly identical distributions over 32 tokens, the unclamped sum in this problem's tests drops as low as $-2 \times 10^{-7}$, and that noise grows with vocabulary size. Careful comparisons of two 8-bit quantizations, whose true divergences can be that small, compute `log_softmax` and the sum in float64 on the CPU (as Learn AI's own `logit_kl` helper does).
