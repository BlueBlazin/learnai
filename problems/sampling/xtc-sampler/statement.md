Implement XTC ("exclude top choices"), a sampler that works in the opposite direction of standard truncation filters. With a given probability, it removes the most likely tokens from a distribution while keeping the tail. To preserve coherence, the least likely token among those top choices always survives.

## Specification

```python
def xtc_filter(logits, threshold, probability, u): ...
```

The inputs represent a batch of $B$ independent next-token logit rows over a vocabulary of size $V$:

- `logits`: next-token logits for each row. Entries may be `-inf` (a masked token with probability 0), and every row has at least one finite entry. `+inf` and NaN never appear.
- `threshold`: a Python float with $0 < \text{threshold} \le 1$, shared across all rows.
- `probability`: a Python float with $0 \le \text{probability} \le 1$, shared across all rows.
- `u`: per-row uniform random draws in $[0, 1)$, so row $b$ is controlled by `u[b]`.

The function returns filtered logits (not probabilities). Kept entries must keep their exact input values, and removed entries must be exactly `-inf`.

Each row $b \in \{0, \ldots, B - 1\}$ is processed independently. Let $\pi = \operatorname{softmax}(\text{logits}[b])$ be the token probabilities of row $b$, where `-inf` entries contribute probability 0. A token $i$ is a **top choice** if

$$
\pi_i \ge \text{threshold}.
$$

XTC **acts** on row $b$ when both of the following conditions hold:

1. `u[b] < probability`, and
2. row $b$ has **at least two** top choices.

When XTC acts on a row, every top choice is set to `-inf` **except the least likely one**. When XTC does not act, the entire row is returned unchanged. Tokens below `threshold` are always returned unchanged.

Key rules:

- **Choosing the survivor.** The least likely top choice is the top choice with the smallest logit. If multiple top choices tie for that exact smallest logit, the one with the **highest token index** survives and the others are removed. Thus, whenever XTC acts, exactly one top choice survives.
- **Single top choice guard.** If only one token meets `threshold`, removing it would leave only low-probability tokens, so the row is left unchanged. For the same reason, any `threshold > 0.5` has no effect because two probabilities cannot both exceed 0.5.
- **Comparison boundaries.** The threshold check is inclusive (`>=`), so a token with probability equal to `threshold` is a top choice. The probability check is strict (`u[b] < probability`), so `probability = 0` never acts and `probability = 1` acts on every row with at least two top choices.
- **No renormalization.** Probabilities are computed once from the input logits, and the output logits are not renormalized.

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

- **Row 0:** `u[0] = 0.2 < 0.5`, and three tokens reach `threshold = 0.1` (probabilities `0.45`, `0.25`, and `0.15`). The two most likely tokens are set to `-inf`. Token 2 (the least likely top choice) survives alongside tokens 3 and 4, which were below `threshold`. Taking softmax of the returned row yields `[0, 0, 0.5, 0.2667, 0.2333]`.
- **Row 1:** Same logits, but `u[1] = 0.8 >= 0.5`. XTC does not act, so the row is unchanged.
- **Row 2:** `u[2] = 0.3 < 0.5`, but only one token reaches `threshold = 0.1`. Because fewer than two top choices exist, the row is unchanged.

## Constraints

- $1 \le B \le 20000$ and $1 \le V \le 2000$.
- `threshold` lies in $[0.01, 1]$, and finite logits lie in $[-1000, 1000]$.
- Kept entries are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. Removed entries must be exactly `-inf`.
- Float32 rounding cannot flip any test case. Every token probability differs from `threshold` by at least $10^{-4}$, except in exact boundary cases where two equal finite logits and `-inf` elsewhere give probability exactly `0.5`. Any two top choices have either identical logits or logits differing by at least $10^{-3}$. Every `u[b]` differs from `probability` by at least $10^{-4}$, except in statistical tests where `probability` is `0.25` or `0.75` (exact in float32).
- **Statistical tests.** The function is called once on 20,000 identical rows with independent uniform draws. The fraction of rows where the most likely token is removed is checked against `probability` using Hoeffding's inequality. A correct solution fails with probability below $10^{-9}$, and because seeds are fixed, it either always passes or always fails.
- No Python loop over rows or tokens is needed, and no sort is needed.

## Why this matters

XTC (Weidmann, 2024) started as a community sampler in text-generation-webui and has since been adopted by llama.cpp, ExLlamaV2, and mlx-lm. Creative-writing front ends enable it to steer models away from repetitive, predictable phrasing. Unlike temperature or repetition penalties, which affect the entire vocabulary, XTC intervenes only when the model already sees multiple plausible continuations and removes the most obvious ones.

The two guard rails are critical to keeping generations readable. Removing the last remaining top choice leaves only low-probability tail tokens and breaks grammar. Acting on a row with a single top choice blocks tokens the model is certain about, such as a closing quote or the second half of a word. Because XTC triggers randomly, bugs in these guards show up as occasional gibberish in long generations rather than deterministic test failures.
