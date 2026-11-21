Implement one denoising step of DiffusionGemma's block text diffusion: sample a
candidate token for every position of a canvas, accept the most certain positions
while their entropies add up to at most a bound, and replace every other position with
a fresh random token. Unlike the masked diffusion of `diffusion-unmask-step`, nothing
is ever committed: acceptance is decided again from scratch at every step.

## Specification

```python
def step_temperature(n, n_max, t_min, t_max): ...  # -> float
def denoise_step(current, logits, temperature, entropy_bound, u_sample, u_noise): ...
# -> (canvas, accepted, argmax_canvas)
```

**`step_temperature`** returns, as a Python float, the sampling temperature of a step
when `n` steps remain, counting the step itself (the first step has $n = n_\text{max}$):

$$
t = t_\text{min} + (t_\text{max} - t_\text{min}) \cdot \frac{n}{n_\text{max}},
\qquad 1 \le n \le n_\text{max}.
$$

**`denoise_step`** handles one canvas of $L$ positions over a vocabulary of $V$
tokens. `logits[i]` is the denoiser's prediction for position $i$, computed from the
canvas `current`. Randomness is an input: position $i$ has the uniforms `u_sample[i]`
and `u_noise[i]`.

1. **Scale.** $p_i = \operatorname{softmax}(\texttt{logits}_i / \texttt{temperature})$.
2. **Sample** a candidate $x_i$ for every position by inverse CDF, as in
   `categorical-sample`: with $C_{i,k} = \sum_{v \le k} p_{i,v}$,
   $x_i = \min\{k : C_{i,k} > u_i \cdot C_{i,V-1}\}$ for $u_i$ = `u_sample[i]`. The
   inequality is strict, so a token of probability 0 is never drawn.
3. **Entropy** of the distribution that was sampled from, in nats:
   $H_i = -\sum_v p_{i,v} \ln p_{i,v}$, where a token whose probability is 0
   (also one that underflowed to 0 in float32) contributes 0.
4. **Accept.** Rank the positions by $H_i$ ascending; among equal entropies the lower
   index ranks first. Let $S_i$ be the sum of the entropies of the positions ranked
   strictly before position $i$ (so $S = 0$ for the first in rank). Position $i$ is
   accepted exactly when $S_i \le \texttt{entropy\_bound}$.
5. **Write.** `canvas[i]` is $x_i$ if position $i$ is accepted, and otherwise the noise
   token $\lfloor \texttt{u\_noise}[i] \cdot V \rfloor$, with the product taken in
   float32. No token of `current` is copied.
6. **Draft.** `argmax_canvas[i]` $= \operatorname{argmax}_v\, \texttt{logits}_{i,v}$,
   for every position, accepted or not.

Arguments and results:

- `current`: integer tensor, shape `(L,)`, values in $[0, V)$.
- `logits`: float32, shape `(L, V)`, finite, with a unique largest entry in every row.
- `temperature`, `entropy_bound`: Python floats, both $> 0$.
- `u_sample`, `u_noise`: float32, shape `(L,)`, $0 \le u < 1$.
- Returns the tuple `(canvas, accepted, argmax_canvas)`: an integer tensor, a bool
  tensor, and an integer tensor, each of shape `(L,)`.
- No tensor passed in is modified.

Do not draw random numbers yourself, and do not call the kit's `ops.eb_accept`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `current`, `canvas`, `argmax_canvas` | `(L,)` | integer |
| `logits` | `(L, V)` | float32 |
| `u_sample`, `u_noise` | `(L,)` | float32 |
| `accepted` | `(L,)` | bool |

## Example

Four positions, three tokens, 12 of 48 steps remaining on the schedule 0.8 to 0.4:

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

Dividing by 0.5 doubles the logits, which gives the probabilities in the comments. The
candidates are $x = (1, 0, 1, 2)$. The ranking is position 1, position 3 (equal
entropy, higher index), position 2, position 0, with
$S = 0,\ 0.3944,\ 0.7888,\ 1.8285$: the first two are within the bound 0.5. Positions
0 and 2 are renoised to $\lfloor 0.9 \cdot 3 \rfloor = 2$ and $\lfloor 0.4 \cdot 3 \rfloor = 1$.

With `entropy_bound = 0.3` only position 1 is accepted, although its own entropy
exceeds the bound, and the canvas is `[2, 0, 1, 1]`.

## Constraints

- $1 \le L \le 256$, $2 \le V \le 262144$, $0.1 \le \texttt{temperature} \le 2$,
  $10^{-6} \le \texttt{entropy\_bound} \le 1000$, $|\texttt{logits}| \le 2000$.
- All three results are compared exactly, with the definition evaluated in float64 and
  with the output of `_denoising_step` in Hugging Face Transformers. The inputs are
  built so that float32 rounding decides nothing:
  - every threshold $u_i \cdot C_{i,V-1}$ is at least $10^{-3}$ away from every
    cumulative sum, except for `u_sample[i]` $= 0$ and the largest float32 below 1,
    which must give the first and the last token of nonzero probability;
  - every $S_i$ is exactly 0 or at least $10^{-3}$ away from the bound, so the tests
    do not distinguish $\le$ from $<$;
  - the last accepted and the first rejected position in rank have identical logit
    rows (equal entropies) or entropies at least $10^{-3}$ apart; in one test with
    $V = 8$, at least $10^{-5}$ apart;
  - `u_noise[i]` $\cdot V$ is at least $10^{-3}$ away from an integer, except for
    `u_noise[i]` $= 0$ and the largest float32 below 1, which must give $V - 1$.
- `step_temperature` is compared with relative tolerance `1e-9`.
- The position with the lowest entropy is accepted for every bound, and the accepted
  set for a larger bound contains the accepted set for a smaller one.

## Why this matters

The step is why text diffusion can write many tokens per forward pass without the
incoherence of sampling them all independently: positions whose entropies sum to
little are nearly determined, so sampling them together loses little. DiffusionGemma
runs it up to 48 times per canvas of 256 tokens with a bound of 0.1. Late in
denoising most positions have probabilities that are exactly 0 and 1 in float32; an
entropy that turns those into NaN rejects exactly the positions that are settled.
Including a position's own entropy in the sum can accept nothing at all, a step that
only reshuffles noise, and keeping rejected tokens instead of renoising them is the
rule of a masked-diffusion sampler, not of this one.
