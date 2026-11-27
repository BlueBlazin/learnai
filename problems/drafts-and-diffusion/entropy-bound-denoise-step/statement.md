Implement one denoising step of DiffusionGemma's block text diffusion. At each step, you sample a candidate token for every position on a canvas, accept the most certain positions whose entropies fit within a budget, and replace every rejected position with a fresh random token. Unlike the masked diffusion in `diffusion-unmask-step`, no token is permanently committed, so acceptance is decided from scratch at every step.

## Specification

```python
def step_temperature(n, n_max, t_min, t_max): ...  # -> float
def denoise_step(current, logits, temperature, entropy_bound, u_sample, u_noise): ...
# -> (canvas, accepted, argmax_canvas)
```

### `step_temperature`

Return, as a Python `float`, the sampling temperature when `n` steps remain out of `n_max` total steps, counting the current step ($1 \le n \le n_\text{max}$, so the first step has $n = n_\text{max}$). The temperature anneals linearly from `t_max` toward `t_min`:

$$
t = t_\text{min} + (t_\text{max} - t_\text{min}) \cdot \frac{n}{n_\text{max}}
$$

### `denoise_step`

Process one canvas of $L$ positions over a vocabulary of $V$ tokens. `current` holds the token ids in $[0, V)$ that produced `logits`, where `logits[i]` is the denoiser's finite prediction for position $i$ (with a unique largest entry in every row). Both `temperature` and `entropy_bound` are positive Python floats. All randomness is supplied as inputs in $[0, 1)$. Use `u_sample[i]` to draw the candidate token at position $i$, and `u_noise[i]` to draw a replacement noise token if position $i$ is rejected.

Perform the following six steps:

1. **Scale.** Compute the temperature-scaled probabilities for each position $i$:
   $$
   p_i = \operatorname{softmax}(\texttt{logits}_i / \texttt{temperature})
   $$
2. **Sample.** Draw a candidate token $x_i$ for every position using inverse-CDF sampling (as in `categorical-sample`). Let $C_{i,k} = \sum_{v \le k} p_{i,v}$ be the cumulative probabilities and let $u_i = \texttt{u\_sample}[i]$. Choose:
   $$
   x_i = \min\{k : C_{i,k} > u_i \cdot C_{i,V-1}\}
   $$
   Because the inequality is strict, a token with probability 0 is never drawn.
3. **Entropy.** Compute the Shannon entropy of the sampled distribution $p_i$ in nats:
   $$
   H_i = -\sum_{v=0}^{V-1} p_{i,v} \ln p_{i,v}
   $$
   Any token whose probability is 0 (including one that underflowed to 0 in float32) contributes 0 to the sum.
4. **Accept.** Rank all $L$ positions by $H_i$ in ascending order, breaking ties by lower position index first. For each position $i$, let $S_i$ be the sum of the entropies of all positions ranked **strictly before** $i$ (so $S = 0$ for the first-ranked position). Accept position $i$ (setting `accepted[i]` to `True`) if and only if $S_i \le \texttt{entropy\_bound}$.
5. **Write.** Set `canvas[i]` to $x_i$ if position $i$ is accepted. Otherwise, replace it with the noise token $\lfloor \texttt{u\_noise}[i] \cdot V \rfloor$, with the product computed in float32. Never copy tokens from `current`.
6. **Draft.** Set `argmax_canvas[i]` $= \operatorname{argmax}_v\, \texttt{logits}_{i,v}$ for every position, whether accepted or not.

Return `(canvas, accepted, argmax_canvas)`. Do not modify any input tensor, do not draw random numbers yourself, and do not call `ops.eb_accept` from `learnai.kit`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `current`, `canvas`, `argmax_canvas` | `(L,)` | integer |
| `logits` | `(L, V)` | float32 |
| `u_sample`, `u_noise` | `(L,)` | float32 |
| `accepted` | `(L,)` | bool |

## Example

Consider four positions, three tokens, and 12 of 48 steps remaining on a temperature schedule from 0.8 down to 0.4:

```python
t = step_temperature(12, 48, 0.4, 0.8)  # 0.5
current = [0, 1, 2, 0]
logits = [
    [0, 0, 0.5 * log(4 / 3)],  # p = (0.30, 0.30, 0.40)   H = 1.0889
    [0.5 * log(18), 0, 0],  #    p = (0.90, 0.05, 0.05)   H = 0.3944
    [0, 0.5 * log(2), 0],  #     p = (0.25, 0.50, 0.25)   H = 1.0397
    [0.5 * log(18), 0, 0],  #    p = (0.90, 0.05, 0.05)   H = 0.3944
]
u_sample = [0.5, 0.5, 0.7, 0.97]
u_noise = [0.9, 0.1, 0.4, 0.5]

denoise_step(current, logits, t, 0.5, u_sample, u_noise)
# canvas = [2, 0, 1, 2]   accepted = [False, True, False, True]   argmax_canvas = [2, 0, 1, 0]
```

Dividing by `t = 0.5` doubles the logits, yielding the probabilities shown in the comments. Inverse-CDF sampling produces candidates $x = (1, 0, 1, 2)$. Ranking by entropy gives position 1, position 3 (same entropy as position 1, higher index), position 2, and position 0. Their prefix sums of earlier entropies are $S = (0,\ 0.3944,\ 0.7888,\ 1.8285)$, so the first two positions (1 and 3) satisfy $S \le 0.5$ and are accepted. Positions 0 and 2 are rejected and renoised to $\lfloor 0.9 \cdot 3 \rfloor = 2$ and $\lfloor 0.4 \cdot 3 \rfloor = 1$.

![The example's four positions in rank order. A position is accepted when the sum $S$ of the entropies ranked before it is at most the bound.](assets/rank-bound.svg)

With `entropy_bound = 0.3`, only position 1 is accepted (even though its own entropy exceeds 0.3), and the resulting canvas is `[2, 0, 1, 1]`.

## Constraints

- $1 \le L \le 256$, $2 \le V \le 262144$, $0.1 \le \texttt{temperature} \le 2$, $10^{-6} \le \texttt{entropy\_bound} \le 1000$, and $|\texttt{logits}| \le 2000$.
- All three outputs of `denoise_step` are compared exactly against a float64 reference and against `_denoising_step` in Hugging Face Transformers. Test inputs are constructed so float32 rounding never flips a decision:
  - Every threshold $u_i \cdot C_{i,V-1}$ is at least $10^{-3}$ away from every cumulative sum, except when `u_sample[i]` is `0` or the largest float32 below `1` (which must select the first and last token of nonzero probability, respectively).
  - Every $S_i$ is either `0` or at least $10^{-3}$ away from `entropy_bound`, so tests do not distinguish $\le$ from $<$.
  - The last accepted and first rejected positions in rank either have identical logit rows (equal entropies) or entropies at least $10^{-3}$ apart (at least $10^{-5}$ apart in one test with $V = 8$).
  - `u_noise[i] * V` is at least $10^{-3}$ away from any integer, except when `u_noise[i]` is `0` or the largest float32 below `1` (which must give $V - 1$).
- `step_temperature` is checked with relative tolerance `1e-9`.
- The lowest-entropy position is always accepted for any bound, and increasing `entropy_bound` can only grow or preserve the accepted set.

## Why this matters

Entropy-bounded acceptance lets a text-diffusion model commit multiple tokens per forward pass without the incoherence of independent parallel sampling. Positions whose entropies sum to a small value are already nearly determined, so sampling them jointly introduces little error. DiffusionGemma runs this step up to 48 times on a 256-token canvas with an entropy bound of 0.1.

Small implementation bugs fail silently here. Late in denoising, most positions have float32 probabilities that underflow to 0 or saturate to 1, so computing `p * log(p)` produces `NaN` and rejects the exact positions that are already settled. Including a position's own entropy in $S_i$ can accept zero positions and waste a step reshuffling noise. Keeping rejected tokens from `current` instead of renoising them turns the sampler into masked diffusion rather than DiffusionGemma's renoising loop.
