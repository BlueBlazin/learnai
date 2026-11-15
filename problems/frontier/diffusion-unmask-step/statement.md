Implement one reverse step of a masked-diffusion language model, and the schedule
that says how many positions each step settles. The model predicts every masked
position at once; the step draws a candidate token for each, keeps the candidates it
is most confident about, and leaves the rest masked for a later step, when more
context will be known. Which probability counts as "confidence", and which positions
may be touched at all, is where implementations go wrong.

## Specification

```python
def schedule(n_masked, steps): ...  # -> list of ints
def unmask_step(tokens, logits, mask_id, n_unmask, temperature, uniforms): ...  # -> tokens
```

Generation starts from a sequence whose positions to generate all hold the mask
token and runs a fixed number of steps. In each step the model maps the current
sequence to logits for every position, and `unmask_step` commits some of the masked
positions. A committed token is never changed again.

**`schedule(n_masked, steps)`** returns how many positions each step commits: a
list of `steps` Python ints,

$$
c_j = \left\lfloor \frac{n}{T} \right\rfloor + \begin{cases} 1 & \text{if } j < n \bmod T \\ 0 & \text{otherwise,} \end{cases} \qquad j = 0, \ldots, T - 1,
$$

with $n$ = `n_masked` $\ge 0$ and $T$ = `steps` $\ge 1$. The counts differ by at most
one, the larger ones come first, and they add up to $n$: after the last step nothing
is masked.

**`unmask_step`** performs one step. Randomness is an input: position $i$ has its
own uniform number `uniforms[i]`.

1. **Exclude the mask token.** The model never predicts the mask token: in every
   row, treat the logit of `mask_id` as $-\infty$. Call the result $z_i$ for
   position $i$.
2. **Draw a candidate** $x_i$ for every position.
   - If `temperature` is 0: $x_i = \operatorname{argmax}_v z_{i,v}$ (the lowest index
     among equal logits), and `uniforms` is not used.
   - Otherwise, with $q_i = \operatorname{softmax}(z_i / \text{temperature})$ and its
     cumulative sums $C_{i,k} = \sum_{v \le k} q_{i,v}$, by inverse-CDF sampling as
     in `categorical-sample`:
     $x_i = \min\{k : C_{i,k} > u_i \cdot C_{i,V-1}\}$.
3. **Confidence.** With $p_i = \operatorname{softmax}(z_i)$, the model's own
   distribution at temperature 1 whatever `temperature` is, the confidence of
   position $i$ is $p_{i, x_i}$: the probability of the candidate that was drawn.
4. **Commit.** Let $m$ be the number of masked positions (`tokens[i] == mask_id`)
   and $n = \min(\texttt{n\_unmask}, m)$. The $n$ masked positions with the highest
   confidence take their candidates; among equal confidences the lowest index goes
   first. Every other position, masked or not, keeps its token.

- `tokens`: an integer tensor of shape `(L,)`.
- `logits`: a float32 tensor of shape `(L, V)`, finite. The rows of positions that
  are not masked are predictions too, and must not have any effect.
- `mask_id`: an int in $[0, V)$. `n_unmask`: an int $\ge 0$. `temperature`: a float
  $\ge 0$.
- `uniforms`: a float32 tensor of shape `(L,)` with $0 \le u < 1$.
- Returns the new tokens, an integer tensor of shape `(L,)`.

Do not draw random numbers yourself.

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

One step with $V = 5$, `mask_id = 4` (written `M`), `n_unmask = 3`, `temperature = 1`.
The table gives each row's $p_i$ over tokens 0 to 3; the mask token's raw logit is
1.0 in every row and is excluded.

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

Position 2: the cumulative sums are `[0.25, 0.50, 0.90, 1.00]` and $u = 0.95$, so the
draw is token 3, an unlikely one. Its confidence is 0.10, the probability of token 3,
not 0.40, the probability of the favorite the draw did not pick; it ranks last and
stays masked. Ranked by the favorite's probability it would have been committed
instead of position 5.

Position 0 is not masked. The model would now say token 0 there, with probability
0.90, more than any masked position's confidence. It is not a candidate for
anything: the 3 that was committed earlier stays.

## Constraints

- $1 \le L \le 20000$, $2 \le V \le 64$, $0 \le$ `n_unmask`, $0 \le u < 1$, finite
  logits. `temperature` is 0 or in $[0.5, 2]$.
- Tokens are compared with the reference exactly. Float32 rounding cannot decide a
  test: at every masked position $u_i \cdot C_{i,V-1}$ is at least $10^{-4}$ away from
  every cumulative sum, at temperature 0 the two largest logits differ by at least
  $10^{-3}$, and the confidences on the two sides of the cut differ by at least
  $10^{-4}$, except in the cases about ties, where rows are identical and their
  confidences are equal bit for bit.
- **Statistical cases.** One call with 20,000 masked positions that share one row
  of logits commits all of them; the committed tokens are then independent draws
  from $\operatorname{softmax}(z / \text{temperature})$ and are compared with it
  using a finite-sample (Chernoff) bound. A correct solution fails with probability
  below $10^{-9}$, and since the seeds are fixed it either always passes or always
  fails.
- No Python loop over positions is needed.

## Why this matters

A diffusion language model generates a whole block of text in a number of forward
passes that does not grow with its length, which is its entire appeal next to an
autoregressive decoder that needs one pass per token. All of the sequencing an
autoregressive model gets for free from left-to-right order is done by this step
instead: which positions to settle now and which to leave for a pass with more
context. If the step ranks by the wrong probability, it settles unlucky draws early
and the later steps have to build around them. If it lets confident predictions
overwrite committed tokens, the text never stops changing and the prompt itself can
be rewritten. If the schedule's counts do not add up, the output is returned with
mask tokens still in it.
