Apply per-row sampling temperature to a batch of next-token logits. Each row uses its own temperature, including `0` for greedy decoding.

![This problem is Temperature, which divides the logits before the softmax.](assets/map.svg)

## Specification

```python
def apply_temperature(logits, temperature): ...
```

The input `logits` holds `B` rows of next-token scores over a vocabulary of `V` tokens. The vector `temperature` supplies one temperature $T_b = \text{temperature}[b]$ for each row $b$. Each row is processed independently so that taking the softmax of the returned logits produces the sampling distribution for that row.

**Positive temperature ($T_b > 0$).** For each token index $v$ in row $b$, divide its logit by $T_b$:

$$
\text{out}[b, v] = \frac{\text{logits}[b, v]}{T_b}.
$$

A masked logit (`-inf`) stays `-inf`.

**Zero temperature ($T_b = 0$).** The row switches to greedy decoding. Let $v^\ast$ be the index of the largest logit in row $b$. **If multiple entries share the largest value, $v^\ast$ is the lowest index among them.** Then:

$$
\text{out}[b, v] =
\begin{cases}
0 & \text{if } v = v^\ast, \\
-\infty & \text{otherwise.}
\end{cases}
$$

This matches the limit of the positive case. For a row of logits $x$, $\operatorname{softmax}(x / T)$ concentrates all probability mass on the largest logit as $T \to 0$. Writing the greedy row as `0` at $v^\ast$ and `-inf` everywhere else ensures its softmax is exactly one-hot.

Input and output value guarantees:

- `logits` may contain `-inf` for masked tokens, and every row has at least one finite entry. Neither `+inf` nor NaN appears.
- Each entry of `temperature` is either exactly `0` or strictly positive. A single batch may mix both.
- The returned tensor never contains NaN or `+inf`, and every row has at least one finite entry.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `temperature` | `(B,)` | float32 |
| result | `(B, V)` | float32 |

## Example

```python
logits = [[2.0, 1.0, 0.0, -inf], [2.0, 1.0, 0.0, -inf], [1.0, 3.0, 3.0, -1.0]]
temperature = [0.5, 2.0, 0.0]

apply_temperature(logits, temperature)
# [[ 4.0, 2.0,  0.0, -inf],
#  [ 1.0, 0.5,  0.0, -inf],
#  [-inf, 0.0, -inf, -inf]]
```

Rows 0 and 1 start with the same logits, whose unscaled softmax is `[0.6652, 0.2447, 0.0900, 0]`. For row 0, dividing by `0.5` doubles the gaps between logits and sharpens the softmax distribution to `[0.8668, 0.1173, 0.0159, 0]`. For row 1, dividing by `2.0` halves the gaps and flattens the distribution to `[0.5065, 0.3072, 0.1863, 0]`. In both rows, the masked token stays `-inf` and keeps probability `0`.

Row 2 has temperature `0`. Its largest logit is `3.0` at indices 1 and 2. Because ties go to the lowest index, token 1 receives `0.0` and every other entry, including the tied token 2, becomes `-inf`. Dividing this row by `0` instead would yield `[inf, inf, inf, -inf]`, whose softmax is NaN.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 1000$.
- Each temperature is either exactly `0` or in $[0.05, 100]$. Finite logits lie in $[-10^5, 10^5]$.
- Results are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. Infinities must match exactly, and any NaN fails.
- Ties for the largest logit occur in the tests for rows with temperature `0`.
- No Python loop over rows or tokens is needed.

## Why this matters

Every generation request specifies a temperature, and a serving engine processes a whole batch of requests at once. Temperature `0` is the most common setting because users rely on it for reproducible output. However, dividing directly by `0` produces `inf` and NaN, which can corrupt downstream sampling into arbitrary token IDs or crashes.

Handling per-row temperatures also prevents cross-request bugs. If a single request's greedy setting or temperature leaks into neighboring rows in the batch, single-request tests will never catch the failure.
