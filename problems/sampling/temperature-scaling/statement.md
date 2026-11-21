Apply sampling temperature to a batch of next-token logits, where every row has its own
temperature. Dividing by the temperature is one line; the exercise is the temperature
of exactly `0`, which means "greedy" and cannot be computed by dividing.

![This problem is Temperature, which divides the logits before the softmax.](assets/map.svg)

## Specification

```python
def apply_temperature(logits, temperature): ...
```

Row $b$ of `logits` is scaled by its own temperature $T_b$ = `temperature[b]`, so that
$\operatorname{softmax}$ of the result is the distribution to sample from.

**Positive temperature.** For $T_b > 0$:

$$
\text{out}[b, v] = \frac{\text{logits}[b, v]}{T_b}.
$$

A masked logit (`-inf`) stays `-inf`.

**Zero temperature.** For $T_b = 0$ the row becomes greedy. Let
$v^\ast$ be the index of the largest logit of the row; **if several entries share the
largest value, $v^\ast$ is the lowest index among them.** Then

$$
\text{out}[b, v] =
\begin{cases}
0 & \text{if } v = v^\ast, \\
-\infty & \text{otherwise.}
\end{cases}
$$

This is the limit of the positive case: as $T \to 0$, $\operatorname{softmax}(x / T)$
puts all its mass on the largest logit. The result is written so that its softmax is
exactly one-hot.

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (masked); every
  row has at least one finite entry. `+inf` and NaN never appear.
- `temperature`: a float32 tensor of shape `(B,)`. Each entry is either exactly `0` or
  positive. One batch may mix both.
- Returns a float32 tensor of shape `(B, V)`. It never contains NaN or `+inf`, and
  every row has at least one finite entry.

Each row depends only on its own logits and its own temperature.

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

Rows 0 and 1 hold the same logits, whose softmax is `[0.6652, 0.2447, 0.0900, 0]`.
Dividing by 0.5 doubles every gap between logits and sharpens the distribution to
`[0.8668, 0.1173, 0.0159, 0]`; dividing by 2.0 halves the gaps and flattens it to
`[0.5065, 0.3072, 0.1863, 0]`. The masked token has probability 0 at any temperature.

Row 2 has temperature 0. Its largest logit, 3.0, occurs at indices 1 and 2; the lowest
index wins, so token 1 gets logit 0 and every other entry, including the tied token 2,
becomes `-inf`. Dividing that row by its temperature instead would give
`[inf, inf, inf, -inf]`, whose softmax is NaN.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 1000$.
- A temperature is exactly `0` or lies in $[0.05, 100]$. Finite logits lie in
  $[-10^5, 10^5]$.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`. Infinities must match exactly, and a NaN is always
  wrong.
- Ties for the largest logit do occur in the tests, in rows with temperature 0.
- No Python loop over rows or tokens is needed.

## Why this matters

Temperature is the one sampling parameter that every request sets, and a serving
engine applies it to a whole batch at once, where each row belongs to a different
request with its own setting. Temperature 0 is the most common value of all (it is
what people send when they want reproducible output), and it is the one value for
which the formula does not work: `x / 0` produces `inf` and NaN, and a NaN in a row
of logits turns into an arbitrary token id or a crash in the sampler several steps
later. Engines therefore special-case it. The other classic failure is a batch in
which one request's greedy setting leaks into its neighbors, which no single-request
test will ever show.
