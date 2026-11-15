Implement the Kullback-Leibler divergence between two next-token distributions, each
given as a row of logits. This is the number that says how far a quantized, pruned, or
draft model's predictions have moved from the original model's. It has to stay finite
and exact when probabilities underflow, and it has to give the right answer, not NaN,
when tokens are masked out.

## Specification

```python
def kl_from_logits(p_logits, q_logits): ...
```

Let $P = \operatorname{softmax}(\texttt{p\_logits})$ and
$Q = \operatorname{softmax}(\texttt{q\_logits})$ along the last axis. For one row with
$V$ tokens the result is

$$
\operatorname{KL}(P \,\|\, Q) = \sum_{i=1}^{V} t_i ,
\qquad
t_i = P_i \,(\log P_i - \log Q_i),
$$

in nats (natural logarithm). $P$, the first argument, is the reference distribution:
the divergence is not symmetric.

A logit of `-inf` marks a masked token, whose probability is exactly $0$. A token with
a finite logit has a positive probability, however small. The terms $t_i$ involving a
zero probability are **defined** as follows, in this order:

1. If $P_i = 0$ (`p_logits[i]` is `-inf`): $t_i = 0$, whatever $Q_i$ is, including
   $Q_i = 0$.
2. Otherwise, if $Q_i = 0$ (`q_logits[i]` is `-inf` and `p_logits[i]` is finite):
   $t_i = +\infty$, and so the row's result is `+inf`. This holds however small $P_i$
   is, also when $e^{\log P_i}$ underflows to `0.0` in float32.
3. Otherwise both logits are finite and $t_i = P_i(\log P_i - \log Q_i)$. Here
   $\log P_i$ and $\log Q_i$ must be the exact log-probabilities (a token whose
   probability underflows in float32 still has a finite log-probability), while the
   factor $P_i$ may underflow to $0$.

The exact divergence is never negative. Float32 rounding can leave a sum such as
`-1e-9` for two nearly identical distributions: return `0.0` instead, so that every
returned value is $\ge 0$.

- `p_logits`, `q_logits`: float32 tensors of the same shape `(..., V)`, with one to
  three dimensions. Entries are finite or `-inf`; every row of each has at least one
  finite entry. `+inf` and NaN never appear.
- Returns a float32 tensor of shape `(...)`: one value per row, and a 0-dimensional
  tensor for 1-D inputs. Each value is a non-negative number or `+inf`, never NaN.

Write the divergence yourself: do not call a built-in KL divergence or relative entropy
(`torch.nn.functional.kl_div`, `torch.nn.KLDivLoss`, `jax.scipy.special.rel_entr`,
`mlx.nn.losses.kl_div_loss`, and the like). Your framework's `log_softmax` and
`logsumexp` are allowed.

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

The first line is $0.5 \ln\frac{0.5}{1/3} + 2 \cdot 0.25 \ln\frac{0.25}{1/3} =
0.202733 - 0.143841$.

Masked tokens, with `ninf = float("-inf")`:

```python
kl_from_logits([0.0, 0.0, ninf], [0.0, 0.0, 0.0])  # 0.405465   rule 1: ln(3/2)
kl_from_logits([0.0, 0.0, ninf], [0.0, ninf, ninf])  # inf        rule 2 at index 1
kl_from_logits([0.0, ninf, ninf], [0.0, 0.0, ninf])  # 0.693147   rule 1 twice: ln 2
```

In the last line index 2 is masked in both, which is rule 1: it contributes 0. Computed
naively it is $0 \cdot (-\infty - (-\infty))$, a NaN.

Probabilities that underflow:

```python
kl_from_logits([0.0, 0.0], [0.0, -200.0])  # 99.306854
kl_from_logits([0.0, -200.0], [0.0, ninf])  # inf
```

In the first line $Q_2 = e^{-200}$ is not representable in float32, but
$\log Q_2 = -200$ is, and the exact answer is $0.5 \ln 0.5 + 0.5(\ln 0.5 + 200)$. In
the second, $P_2 = e^{-200}$ is positive although it rounds to `0.0`, and $Q_2 = 0$:
rule 2.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. Infinities must match exactly, a NaN is always wrong,
  and so is any value below `0`.
- Finite logits lie in $[-10^4, 10^4]$, and $1 \le V \le 4096$. The tolerance applies at
  every magnitude, also for rows whose logits all sit near $10^4$, where float32
  resolves steps of about $10^{-3}$.
- The inputs have at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32.

## Why this matters

When you quantize weights to 4 bits, switch a model to half precision, or pick a draft
model for speculative decoding, the question is always how much the output distribution
changed, and the standard answer is the mean KL divergence between the original
model's logits and the new model's, token position by token position. The same number
predicts the acceptance rate of a speculative draft. Evaluation harnesses run it over
logits that contain masked tokens (constrained decoding, banned tokens) and very
confident predictions, which is exactly where a version built on probabilities returns
NaN or infinity for one position and turns the mean over a whole dataset into NaN.
