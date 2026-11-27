Implement inverse-CDF sampling to draw one token index per row from a batch of
probability distributions using caller-supplied uniform random numbers. Your
implementation must never select a zero-probability token and must always return a
valid index even when row probabilities do not sum to exactly 1.

![This problem is Sample, which turns a distribution and one uniform number into a token.](assets/map.svg)

## Specification

```python
def sample(probs, u): ...
```

The input `probs` holds `B` probability distributions over a vocabulary of `V`
tokens. All entries are non-negative and may be exactly 0. Every row has a positive
total close to 1, ranging between 0.999 and 1.001, because float32 probabilities
rarely sum to 1 exactly.

Randomness is supplied as an input so the function is deterministic. The vector `u`
contains one uniform random number per row, with $0 \le u[b] < 1$ for row $b$.

For a single row with probabilities $p_0, \ldots, p_{V-1}$ and uniform number $u$,
compute the cumulative sums

$$
c_k = \sum_{i=0}^{k} p_i
$$

where the last entry $c_{V-1}$ is the row's total. The sampled token is the first
index whose cumulative sum strictly exceeds $u$ scaled by that row total:

$$
\text{token} = \min \{\, k : c_k > u \cdot c_{V-1} \,\}.
$$

Scaling $u$ by $c_{V-1}$ is equivalent to normalizing the row first, without
dividing every entry.

The following properties follow from this definition and are verified by the tests:

- **Zero-probability tokens are never returned.** A token with $p_k = 0$ has
  $c_k = c_{k-1}$, so it can never be the first index to exceed the threshold. In
  particular, `u = 0` returns the first token with positive probability, not token 0
  when $p_0 = 0$.
- **The result is always a valid index in $[0, V)$.** Because $u < 1$, the threshold
  $u \cdot c_{V-1}$ is strictly less than $c_{V-1}$. Even if a row sums to 0.999 and
  `u = 0.9995`, some $c_k$ exceeds the threshold, and trailing zero-probability
  tokens are never returned as a fallback.
- **Ties advance to the next positive token.** If $u \cdot c_{V-1}$ equals $c_k$
  exactly, token $k$ is not selected. The result is the next token with positive
  probability.
- **Correct distribution.** When `u` is uniform on $[0, 1)$, token $k$ is returned
  with probability $p_k / c_{V-1}$.

Do not draw random numbers yourself, and do not call your framework's categorical or
multinomial sampler.

## Shapes

| name | shape | dtype |
|---|---|---|
| `probs` | `(B, V)` | float32 |
| `u` | `(B,)` | float32 |
| result | `(B,)` | integer |

## Example

```python
probs = [[0.1, 0.4, 0.2, 0.3], [0.0, 0.5, 0.0, 0.5], [0.25, 0.25, 0.499, 0.0]]
u = [0.55, 0.0, 0.9995]

sample(probs, u)  # [2, 1, 2]
```

In row 0, the cumulative sums are `[0.1, 0.5, 0.7, 1.0]` and the threshold is
$0.55 \times 1.0 = 0.55$. The first sum strictly above 0.55 is $c_2 = 0.7$, so the
result is token 2.

In row 1, the cumulative sums are `[0.0, 0.5, 0.5, 1.0]` and the threshold is 0.
Since $c_0 = 0$ does not exceed 0, token 0 (which has probability 0) is skipped. The
first sum above 0 is $c_1 = 0.5$, so the result is token 1.

![Rows 0 and 1 as unit intervals cut at the cumulative sums. Tokens 0 and 2 of row 1 have zero width, so no $u$ selects them.](assets/unit-interval.svg)

In row 2, the cumulative sums are `[0.25, 0.5, 0.999, 0.999]`, so the row total is
0.999. Comparing directly against the raw `u = 0.9995` would find no sum above it.
Scaling by the row total gives threshold $0.9995 \times 0.999 = 0.9985$, and the
first sum above it is $c_2 = 0.999$, giving token 2. Token 3 has probability 0 and
is not a valid fallback.

## Constraints

- $1 \le B \le 4096$, $1 \le V \le 1000$, and $0 \le u < 1$. Each row's total lies
  in $[0.999, 1.001]$.
- Outputs are compared with the reference for exact equality. To avoid float32
  boundary ambiguity, every threshold $u \cdot c_{V-1}$ differs from every
  cumulative sum $c_k$ by at least $10^{-4} \cdot c_{V-1}$. The only exceptions are
  inputs whose arithmetic is exact in float32 (`u = 0`, rows whose entries are
  multiples of 1/8, and one-hot rows), which test the strict inequality.
- **Statistical cases.** The function is called once with 4000 identical rows and
  4000 independent uniform values. The empirical token distribution is checked
  against the row's normalized probabilities using a finite-sample (Chernoff) bound.
  A correct solution fails with probability below $10^{-9}$, and fixed seeds make
  the check deterministic.
- No Python loop over rows or tokens is needed.

## Why this matters

Inverse-CDF sampling is the final step of generating a token, turning a filtered
probability distribution and one random number into a token ID. In float32, a
cumulative sum of probabilities ends at 0.99999994 or 1.0000001 as often as at 1.
Distributions renormalized after top-p filtering or computed in half precision can
drift even further.

Code that assumes the last cumulative sum is 1 occasionally produces index `V`, an
out-of-range token ID that crashes the embedding lookup on the next step once in
millions of tokens. Common fixes trade that crash for a silent bug by clamping to
the last token or falling back to token 0, which can emit a token that a filter or
grammar had just excluded.
