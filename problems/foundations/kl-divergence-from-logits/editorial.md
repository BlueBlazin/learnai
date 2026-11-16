## The idea

$\operatorname{KL}(P \,\|\, Q) = \sum_i P_i (\log P_i - \log Q_i)$ is the expected
number of extra nats you pay for coding samples of $P$ with a code built for $Q$. In
model evaluation $P$ is the model you trust (full precision, the target model) and $Q$
the one you are testing (quantized, half precision, a draft). The order matters: the
sum is weighted by $P$, so it measures the damage on the tokens the original model
actually predicts.

Three things go wrong when this is computed in floating point.

**Probabilities underflow, log-probabilities do not.** A float32 probability below
about $10^{-45}$ is `0.0`, and its logarithm is `-inf`. The log-probability from
`log_softmax` is just a moderately large negative number. So both logarithms must
come from `log_softmax`, never from `log(softmax(...))`. The factor $P_i$ in front may
underflow freely: $P_i \cdot (\text{finite})$ is then $0$, which is the correctly
rounded value of a term smaller than $10^{-40}$.

**Zero probabilities need a definition.** With masked tokens the formula produces
$0 \cdot (-\infty)$ and $-\infty - (-\infty)$, both NaN. The mathematical convention
is fixed:

- $0 \log (0 / q) = 0$, because $x \log x \to 0$ as $x \to 0$. A token $P$ never
  produces costs nothing, whatever $Q$ says about it.
- $p \log (p / 0) = +\infty$ for $p > 0$. $Q$ declares impossible something that $P$
  does: no finite penalty describes that.

The subtle part is *what counts as zero*. A masked logit is an exact zero. A finite
logit is a positive probability, even at $e^{-200}$. Testing `p == 0` on the
exponentiated value confuses the two and returns a finite divergence where the answer
is infinite. Decide from the logits.

**Rounding can cross zero.** By Gibbs' inequality the exact sum is $\ge 0$, with
equality only for $P = Q$. The individual terms have both signs, though, and for
nearly identical distributions they cancel down to a sum the size of float32 rounding
error, which comes out negative about half the time. Clamping at 0 costs nothing and
keeps a later `sqrt`, a log-scale plot, or a `>= 0` assertion from failing.

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

The unconditional product contains NaN exactly at entries that one of the two `where`
calls overwrites, so no NaN reaches the sum. The order of the two calls implements the
order of the rules.

The JAX version is the same with `jax.nn.log_softmax` and `jnp.where`. MLX has no
`log_softmax` in `mlx.core`, and the tempting `x - mx.logsumexp(x, axis=-1,
keepdims=True)` is not good enough: for a row of logits near $10^4$ the log-sum-exp is
itself near $10^4$, where float32 resolves steps of $10^{-3}$, and the difference
inherits that error. The MLX reference shifts first, as in `stable-softmax`:

```python
shifted = x - mx.max(x, axis=-1, keepdims=True)
log_p = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
```

## Common mistakes

- **Swapping the arguments.** `torch.nn.functional.kl_div(input, target)` takes the
  *approximation* first, as log-probabilities, and the reference second, which is the
  opposite of the mathematical notation. Half of the swapped-KL bugs in evaluation
  code come from that signature.
- **No rule for zeros.** One masked token in $P$ turns the row into NaN, and a mean
  over a dataset into NaN.
- **Working on probabilities.** `log(softmax(q))` is `-inf` for a confident model's
  unlikely tokens. The row becomes `+inf` although its divergence is, say, 99.3 nats.
- **Deciding $P_i = 0$ from the underflowed value.** Correct for masked tokens, wrong
  for a finite logit that $Q$ masks: the row must be infinite.
- **Skipping tokens masked in $Q$.** It looks like robust handling of `-inf` and
  silently reports a perfect score for a model that forbids the right token.
- **`x - logsumexp(x)` without the shift.** Correct for logits of order 1, off by up
  to $10^{-3}$ nats per row for logits of order $10^4$, which is larger than the
  divergence between a model and its 8-bit quantization.
- **Returning the raw sum.** Slightly negative divergences for nearly identical
  distributions.
- **Reducing over the batch.** `reduction="batchmean"` habits return one number; the
  caller here wants one value per position and takes its own mean, usually over
  non-padding positions only.

## Complexity

Two `log_softmax` passes and a few elementwise operations: $O(V)$ time and memory per
row.

## In the wild

llama.cpp's `perplexity` tool has a `--kl-divergence` mode: logits of the original
model are saved with `--kl-divergence-base`, and a quantized model is then compared
against them, with the mean KL divergence per token as the headline statistic. In speculative decoding the acceptance probability of a draft
token is $\min(1, P_i / Q_i)$ (Leviathan et al., 2023), and the total variation
distance that governs the acceptance rate is bounded by $\sqrt{\mathrm{KL}/2}$
(Pinsker's inequality), which is why a draft model is first judged by this number.

The float32 sum carries rounding noise: for two nearly identical distributions over
32 tokens the unclamped result in this problem's tests is as low as $-2 \times
10^{-7}$, and the noise grows with the vocabulary. A careful comparison of two 8-bit
quantizations, whose divergences can be that small, computes the log-softmax and the
sum in float64 on the CPU, as Learn AI's own `logit_kl` helper does.
