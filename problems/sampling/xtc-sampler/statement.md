Implement XTC ("exclude top choices"), a sampler that does the opposite of every
truncation filter: with some probability it removes the *most* likely tokens of a row
and keeps the tail. What keeps the output coherent is one rule that is easy to get
wrong: of the tokens that qualify as top choices, the least likely one always
survives.

## Specification

```python
def xtc_filter(logits, threshold, probability, u): ...
```

Randomness is an input: row $b$ is decided by its own uniform number `u[b]`.

Each row is treated independently. For one row, let
$\pi = \operatorname{softmax}(\text{logits})$. The **top choices** are the tokens with

$$
\pi_i \;\ge\; \text{threshold}.
$$

XTC **acts** on the row when both conditions hold:

1. `u[b] < probability`, and
2. the row has **at least two** top choices.

When it acts, every top choice is removed (set to `-inf`) **except the least likely
one**. When it does not act, the row is returned unchanged. In both cases the tokens
below the threshold are returned unchanged.

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (a masked token,
  probability 0); every row has at least one finite entry. `+inf` and NaN never appear.
- `threshold`: a Python float with $0 < \text{threshold} \le 1$, the same for every
  row.
- `probability`: a Python float with $0 \le \text{probability} \le 1$, the same for
  every row.
- `u`: a float32 tensor of shape `(B,)` with $0 \le u < 1$.
- Returns a float32 tensor of shape `(B, V)`: logits, not probabilities. A kept entry
  has exactly its input value; a removed entry is exactly `-inf`.

Details that decide the output:

- **The survivor.** The least likely top choice is the one with the smallest logit.
  If several top choices share that smallest logit exactly, the one with the
  **highest index** survives and the others are removed. So when XTC acts, exactly
  one top choice is left.
- **A single top choice is never removed.** If only one token reaches the threshold,
  removing it would leave only what the model considers unlikely; the row is
  unchanged. For the same reason a threshold above 0.5 can never have an effect: two
  probabilities cannot both exceed one half.
- **The comparisons.** A token at exactly the threshold is a top choice (`>=`). The
  row is filtered when `u[b] < probability` (strictly), so `probability = 0` never
  acts and `probability = 1` acts on every row with two top choices.
- The probabilities are those of the logits as given (the softmax over the whole
  row, masked entries contributing 0). Nothing is renormalized afterwards.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `u` | `(B,)` | float32 |
| result | `(B, V)` | float32 |

## Example

```python
# softmax(logits) = [[0.45, 0.25, 0.15, 0.08, 0.07],
#                    [0.45, 0.25, 0.15, 0.08, 0.07],
#                    [0.92, 0.05, 0.02, 0.01, 0.00]]
logits = [
    [-0.7985, -1.3863, -1.8971, -2.5257, -2.6593],
    [-0.7985, -1.3863, -1.8971, -2.5257, -2.6593],
    [-0.0834, -2.9957, -3.9120, -4.6052, -inf],
]
u = [0.2, 0.8, 0.3]

xtc_filter(logits, 0.1, 0.5, u)
# [[   -inf,    -inf, -1.8971, -2.5257, -2.6593],
#  [-0.7985, -1.3863, -1.8971, -2.5257, -2.6593],
#  [-0.0834, -2.9957, -3.9120, -4.6052,    -inf]]
```

Row 0: `u = 0.2 < 0.5`, and three tokens reach the threshold of 0.1 (0.45, 0.25,
0.15). The two most likely are removed; token 2, the least likely top choice,
survives, and so do tokens 3 and 4, which were never top choices. Sampling from the
result means taking its softmax: `[0, 0, 0.5, 0.2667, 0.2333]`.

Row 1: the same logits with `u = 0.8`. XTC does not act.

Row 2: `u = 0.3 < 0.5`, but only one token reaches the threshold. The model is sure
of this token, and XTC leaves the row alone.

## Constraints

- $1 \le B \le 20000$, $1 \le V \le 2000$. `threshold` lies in $[0.01, 1]$. Finite
  logits lie in $[-1000, 1000]$.
- Kept entries are compared with the reference to a relative tolerance of `1e-4` and
  an absolute tolerance of `1e-6` (they are copies of the input); removed entries must
  be exactly `-inf`.
- Float32 rounding cannot decide a test. Every probability differs from the threshold
  by at least $10^{-4}$, except where both are exact: two equal finite logits in a
  row whose other entries are `-inf` have probability exactly 0.5. Two top choices
  have either exactly equal logits or logits that differ by at least $10^{-3}$. Every
  `u[b]` differs from `probability` by at least $10^{-4}$, except in the statistical
  cases, where `probability` is 0.25 or 0.75, which float32 represents exactly.
- **Statistical cases.** The function is called once with 20,000 identical rows and
  independent uniforms, and the fraction of rows in which the most likely token was
  removed is compared with `probability` using Hoeffding's inequality. A correct
  solution fails with probability below $10^{-9}$, and since the seeds are fixed, it
  either always passes or always fails.
- No Python loop over rows or tokens is needed, and no sort.

## Why this matters

XTC (Weidmann, 2024; first a community sampler in text-generation-webui, since
adopted by llama.cpp, ExLlamaV2, and mlx-lm) is what creative-writing front ends turn
on to break a model out of its most predictable phrasing. Repetition penalties and
temperature act on all tokens; XTC acts only where the model has several acceptable
continuations and removes the obvious ones. The guard rails are the whole design.
Remove the last top choice too, and the sampler falls through to tokens the model
gave a percent or less: grammar breaks. Act on a row with a single top choice, and
the one token that was certainly right (the closing quote, the second half of a
word) becomes impossible. Because the sampler fires at random, such bugs show up as
occasional garbage in long generations, not as a failing prompt.
