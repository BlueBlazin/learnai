Implement the Kullback-Leibler (KL) divergence between two next-token probability distributions, each given as a row of unnormalized logits. This metric measures how far a quantized, pruned, or draft model's predictions have drifted from the original model's. Your implementation must stay finite and accurate when probabilities underflow, and it must handle masked tokens without producing NaN.

## Specification

```python
def kl_from_logits(p_logits, q_logits): ...
```

Let $P = \operatorname{softmax}(\texttt{p\_logits})$ and $Q = \operatorname{softmax}(\texttt{q\_logits})$ along the last axis. For one row over a vocabulary of $V$ tokens, the KL divergence in nats (using the natural logarithm) is

$$
\operatorname{KL}(P \,\|\, Q) = \sum_{i=1}^{V} t_i ,
\qquad
t_i = P_i \,(\log P_i - \log Q_i).
$$

Here $P$ (from the first argument) is the reference distribution, and $Q$ is the approximating distribution. The divergence is not symmetric, so the argument order matters.

A logit of `-inf` marks a masked token whose probability is exactly $0$. Any token with a finite logit has a strictly positive probability, however small. Define each term $t_i$ in the following priority order:

1. If $P_i = 0$ (`p_logits[i]` is `-inf`), then $t_i = 0$ regardless of $Q_i$, including when $Q_i = 0$.
2. Otherwise, if $Q_i = 0$ (`q_logits[i]` is `-inf` while `p_logits[i]` is finite), then $t_i = +\infty$, so the row's divergence is `+inf`. This holds no matter how small $P_i$ is, even when $e^{\log P_i}$ underflows to `0.0` in float32.
3. Otherwise, both logits are finite and $t_i = P_i(\log P_i - \log Q_i)$. Here $\log P_i$ and $\log Q_i$ must be exact log-probabilities (a token whose probability underflows in float32 still has a finite log-probability), while the leading factor $P_i$ may underflow to `0.0`.

The exact KL divergence is never negative. Because float32 rounding can leave a tiny negative sum such as `-1e-9` for two nearly identical distributions, clamp negative sums to `0.0` so every returned value is $\ge 0$.

Entries of `p_logits` and `q_logits` are either finite or `-inf`, and every row of each tensor contains at least one finite entry (`+inf` and NaN never appear). For 1-D inputs, return a 0-dimensional float32 tensor. Every returned value must be a non-negative float or `+inf`, never NaN.

Write the divergence yourself. Do not call a built-in KL divergence or relative entropy function (`torch.nn.functional.kl_div`, `torch.nn.KLDivLoss`, `jax.scipy.special.rel_entr`, `mlx.nn.losses.kl_div_loss`, and similar). Your framework's `log_softmax` and `logsumexp` are allowed.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p_logits` | `(..., V)`, one to three dimensions | float32 |
| `q_logits` | same as `p_logits` | float32 |
| result | `(...)` | float32 |

## Example

```python
p = [0.693147, 0.0, 0.0]  # P = [0.5, 0.25, 0.25]
q = [0.0, 0.0, 0.0]  # Q = [1/3, 1/3, 1/3]

kl_from_logits(p, q)  # 0.058892
kl_from_logits(q, p)  # 0.056633   not symmetric
kl_from_logits(p, p)  # 0.0
```

The first line evaluates $0.5 \ln\frac{0.5}{1/3} + 2 \cdot 0.25 \ln\frac{0.25}{1/3} = 0.202733 - 0.143841$.

With masked tokens (`ninf = float("-inf")`):

```python
kl_from_logits([0.0, 0.0, ninf], [0.0, 0.0, 0.0])  # 0.405465   rule 1: ln(3/2)
kl_from_logits([0.0, 0.0, ninf], [0.0, ninf, ninf])  # inf        rule 2 at index 1
kl_from_logits([0.0, ninf, ninf], [0.0, 0.0, ninf])  # 0.693147   rule 1 twice: ln 2
```

In the last line, index 2 is masked in both distributions. By rule 1 it contributes `0`, whereas evaluating $0 \cdot (-\infty - (-\infty))$ directly would produce NaN.

With probabilities that underflow in float32:

```python
kl_from_logits([0.0, 0.0], [0.0, -200.0])  # 99.306854
kl_from_logits([0.0, -200.0], [0.0, ninf])  # inf
```

In the first line, $Q_2 = e^{-200}$ underflows in float32, but $\log Q_2 = -200$ is representable, giving the exact answer $0.5 \ln 0.5 + 0.5(\ln 0.5 + 200) = 99.306854$. In the second line, $P_2 = e^{-200}$ is strictly positive even though it rounds to `0.0`, while $Q_2 = 0$, so rule 2 gives `+inf`.

## Constraints

- Results are compared against the reference with relative tolerance `1e-4` and absolute tolerance `1e-5`. Infinities must match exactly, and any NaN or value below `0` is always wrong.
- Finite logits lie in $[-10^4, 10^4]$, and $1 \le V \le 4096$. The tolerance applies at every magnitude, including rows whose logits all sit near $10^4$ (where float32 spacing is about $10^{-3}$).
- The inputs have at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32.

## Why this matters

When you quantize weights to 4 bits, cast a model to half precision, or choose a draft model for speculative decoding, you need to know how much the output distribution changed. The standard metric is the mean KL divergence between the original model's logits and the new model's logits across token positions, and the same quantity bounds the acceptance rate of a speculative draft. Evaluation harnesses run this metric over logits with masked tokens (from constrained decoding or banned tokens) and sharp, high-confidence predictions. A naive implementation built on probabilities returns NaN or `+inf` on a single position and corrupts the mean over the entire dataset.
