Implement one reverse unmasking step of a masked-diffusion language model, along with the schedule that determines how many positions each step commits. At each step, the model predicts a candidate token for every masked position, commits the most confident ones, and leaves the rest masked for later steps when more context is available.

## Specification

```python
def schedule(n_masked, steps): ...  # -> list of ints
def unmask_step(tokens, logits, mask_id, n_unmask, temperature, uniforms): ...  # -> tokens
```

Generation starts from a sequence where the positions to generate hold the mask token and runs for a fixed number of steps. In each step, the model produces logits for every position, and `unmask_step` commits some of the masked positions. Once a position is committed, its token never changes.

**`schedule(n_masked, steps)`** returns a list of `steps` Python `int`s specifying how many positions each step commits. Let $n = \texttt{n\_masked} \ge 0$ and $T = \texttt{steps} \ge 1$. The schedule splits $n$ as evenly as possible across $T$ steps, placing the remainder in the earliest steps:

$$
c_j = \left\lfloor \frac{n}{T} \right\rfloor + \begin{cases} 1 & \text{if } j < n \bmod T \\ 0 & \text{otherwise,} \end{cases} \qquad j = 0, \ldots, T - 1.
$$

The counts differ by at most one, the larger counts come first, and their sum is $n$ so that no masked positions remain after the last step.

**`unmask_step`** performs one reverse step on a sequence of length $L$ over a vocabulary of size $V$:

- `tokens` holds the current token IDs at each position $i \in \{0, \ldots, L - 1\}$.
- `logits` holds finite model logits for every position and vocabulary token. Rows for positions that are already unmasked are still present in `logits` and must have no effect on the output.
- `mask_id` is an `int` in $[0, V)$ identifying the mask token.
- `n_unmask` is an `int` $\ge 0$ giving the target number of masked positions to commit.
- `temperature` is a `float` $\ge 0$ controlling sampling randomness.
- `uniforms` supplies all randomness, with position $i$ using its own uniform draw $u_i = \texttt{uniforms}[i] \in [0, 1)$. Do not draw random numbers yourself.

Compute the updated token tensor in four stages:

1. **Exclude the mask token.** The model never outputs the mask token. In every row $i$, replace the logit at index `mask_id` with $-\infty$, and denote the resulting logit vector by $z_i$.
2. **Draw a candidate token $x_i$ for each position $i$.**
   - If `temperature` is 0, set $x_i = \operatorname{argmax}_v z_{i,v}$, breaking ties in favor of the lowest token index. Do not use `uniforms`.
   - Otherwise, compute the tempered distribution $q_i = \operatorname{softmax}(z_i / \text{temperature})$ and its cumulative sums $C_{i,k} = \sum_{v \le k} q_{i,v}$. Draw $x_i$ by inverse-CDF sampling (as in `categorical-sample`):
     $$x_i = \min\{k : C_{i,k} > u_i \cdot C_{i,V-1}\}.$$
3. **Compute confidence.** Let $p_i = \operatorname{softmax}(z_i)$ be the model's unscaled distribution at temperature 1, regardless of `temperature`. The confidence of position $i$ is $p_{i, x_i}$, the unscaled probability of the drawn candidate $x_i$.
4. **Commit the top masked positions.** Let $m$ be the number of currently masked positions (`tokens[i] == mask_id`), and let $n = \min(\texttt{n\_unmask}, m)$. Select the $n$ masked positions with the highest confidence, breaking ties in favor of the lowest position index. Replace `tokens[i]` with $x_i$ at those $n$ positions, and leave every other position (masked or unmasked) unchanged. Return the resulting token tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(L,)` | integer |
| `logits` | `(L, V)` | float32 |
| `uniforms` | `(L,)` | float32 |
| result of `unmask_step` | `(L,)` | integer |
| result of `schedule` | list of `steps` ints | Python `int` |

## Example

```python
schedule(10, 4)  # [3, 3, 2, 2]
schedule(3, 5)  # [1, 1, 1, 0, 0]
```

Now consider one unmasking step with $V = 5$, `mask_id = 4` (written `M`), `n_unmask = 3`, and `temperature = 1.0`. The table below shows each row's unscaled probabilities $p_i$ over tokens 0 through 3 after excluding the mask token (whose raw logit is 1.0 in every row).

```python
tokens = [3, M, M, 1, M, M]
uniforms = [0.2, 0.5, 0.95, 0.4, 0.3, 0.6]

unmask_step(tokens, logits, 4, 3, 1.0, uniforms)  # [3, 0, M, 1, 1, 1]
```

| $i$ | token | $p_i$ | $u_i$ | candidate | confidence | outcome |
|---|---|---|---|---|---|---|
| 0 | 3 | `[0.90, 0.04, 0.03, 0.03]` | 0.2 | 0 | (not masked) | stays 3 |
| 1 | `M` | `[0.70, 0.10, 0.10, 0.10]` | 0.5 | 0 | 0.70 | committed: 0 |
| 2 | `M` | `[0.25, 0.25, 0.40, 0.10]` | 0.95 | 3 | 0.10 | stays masked |
| 3 | 1 | `[0.25, 0.25, 0.25, 0.25]` | 0.4 | 1 | (not masked) | stays 1 |
| 4 | `M` | `[0.10, 0.60, 0.20, 0.10]` | 0.3 | 1 | 0.60 | committed: 1 |
| 5 | `M` | `[0.50, 0.30, 0.10, 0.10]` | 0.6 | 1 | 0.30 | committed: 1 |

![One step of the example. Four masked positions each draw a candidate, and the three most confident commit.](assets/unmask-steps.svg)

At position 2, the cumulative sums are `[0.25, 0.50, 0.90, 1.00]` and $u_2 = 0.95$, so the draw selects token 3. Its confidence is 0.10 (the probability of token 3), not 0.40 (the probability of token 2, which was not drawn). Because 0.10 is the lowest confidence among the four masked positions, position 2 stays masked. Ranking by the maximum probability instead would have committed position 2 over position 5.

At position 0, the model assigns probability 0.90 to token 0, which is higher than any masked position's confidence. Because position 0 is already unmasked, it is ineligible and keeps token 3.

## Constraints

- $1 \le L \le 20000$, $2 \le V \le 64$, `n_unmask` $\ge 0$, $0 \le u_i < 1$, and all logits are finite.
- `temperature` is either 0 or in $[0.5, 2]$.
- Output tokens are compared against the reference for exact equality. Test inputs are constructed so float32 rounding cannot flip a decision. At every masked position, $u_i \cdot C_{i,V-1}$ is at least $10^{-4}$ away from every cumulative sum. At temperature 0, the two largest logits differ by at least $10^{-3}$. Confidences on opposite sides of the commit cutoff differ by at least $10^{-4}$, except in tie-breaking tests where rows are identical and their confidences are bit-for-bit equal.
- **Statistical tests.** One test passes 20,000 masked positions sharing the same logit row and commits all of them. The committed tokens form independent draws from $\operatorname{softmax}(z / \text{temperature})$ and are checked against it with a finite-sample Chernoff bound. A correct solution fails with probability below $10^{-9}$, and because the seeds are fixed, it either always passes or always fails.
- No Python loop over positions is needed.

## Why this matters

A masked-diffusion language model generates an entire block of text in a fixed number of forward passes rather than one pass per token like an autoregressive decoder. Because it lacks left-to-right ordering, this unmasking step controls the generation order by deciding which positions to settle now and which to defer until more context is known.

Small mistakes in this step silently degrade generation. Ranking by the wrong probability commits unlikely draws early and forces later steps to build around them. Allowing confident predictions to overwrite unmasked positions corrupts already-committed text and can even rewrite the prompt. A schedule whose counts do not sum to `n_masked` leaves mask tokens in the final output.
