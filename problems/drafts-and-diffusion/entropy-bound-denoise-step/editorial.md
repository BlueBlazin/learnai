## The idea

A text-diffusion model predicts every position of a canvas in one pass, but its
prediction is a product of independent per-position distributions. Sampling all
positions at once from that product ignores how they depend on each other: two
positions that are each "either *New York* or *Los Angeles*" can come out as *New
Angeles*. Sampling one position per pass is safe and no faster than an autoregressive
model.

Entropy-bounded acceptance sits in between. A position with entropy near zero is
already determined, so nothing is lost by fixing it together with others. The sampler
takes positions in order of increasing entropy and stops when the entropies of the
ones already taken add up to more than a budget. The sum that is compared with the
bound for a position is the entropy of everything accepted **before** it, which has
two consequences:

- the first position in rank is compared with 0 and is always accepted, so every step
  makes progress, even when the most certain position is still quite uncertain;
- the position that crosses the budget is itself accepted. In the library's words the
  rule accepts the $k$ lowest-entropy tokens with
  $\sum_{i \le k} H_i - \max_{i \le k} H_i \le \texttt{entropy\_bound}$: the sum
  without the largest term, which in ascending order is the last one.

What happens to the rest is what separates this sampler from masked diffusion. The
canvas starts as uniformly random tokens, there is no mask token, and a rejected
position is **renoised**: it gets a new uniformly random token, not its old one.
Acceptance is recomputed from the new logits at every step. A position accepted at one
step can be rejected and renoised at the next. The token ids of `current` reach the new
canvas only through the logits the denoiser computed from them, which is why `current`
is an argument that a correct solution never reads.

The temperature falls linearly from $t_\text{max}$ to almost $t_\text{min}$ as the
steps run out. It is applied before both the sampling and the entropy, so a lower
temperature also makes more positions pass the bound. `argmax_canvas` is the running
draft: the library streams it, tests it for convergence, and appends it (not the
sampled canvas) to the text when the canvas is finished. Dividing by a positive
temperature does not change an argmax.

## Reference solution (PyTorch)

```python
def step_temperature(n, n_max, t_min, t_max):
    return t_min + (t_max - t_min) * (n / n_max)


def denoise_step(current, logits, temperature, entropy_bound, u_sample, u_noise):
    vocab = logits.shape[-1]
    log_p = torch.log_softmax(logits / temperature, dim=-1)
    p = torch.exp(log_p)

    cdf = torch.cumsum(p, dim=-1)
    sampled = (cdf <= u_sample[:, None] * cdf[:, -1:]).sum(dim=-1)

    entropy = -(p * log_p).sum(dim=-1)

    order = torch.argsort(entropy, stable=True)
    ranked = entropy[order]
    before = torch.cumsum(ranked, dim=0) - ranked
    accepted = (before <= entropy_bound)[torch.argsort(order)]

    noise = torch.floor(u_noise * vocab).to(torch.int64)
    canvas = torch.where(accepted, sampled, noise)
    return canvas, accepted, torch.argmax(logits, dim=-1)
```

`argsort(order)` is the inverse permutation: `order[r]` is the position at rank `r`,
and `argsort(order)[i]` is the rank of position `i`.

## Numerics

- **Zero probabilities.** At temperature 0.4, a logit gap of 42 is a gap of 105 after
  the division, and `exp(-105)` is 0 in float32. `log_softmax` stays finite there, so
  `p * log_p` is `0 * (-105) = 0`. Taking `log(p)` instead gives `0 * -inf = NaN`, the
  NaN poisons the row's entropy, and the most settled positions of the canvas are the
  ones that get rejected.
- **The noise token.** $\lfloor u V \rfloor < V$ needs $uV < V$ after rounding. The
  largest float32 below 1 is $1 - 2^{-24}$, and $V(1 - 2^{-24})$ rounds to a float32
  below $V$ for every $V$ that was tried (all up to 70000, and $2^{18}$, $2^{24}$), so
  no clamp is needed, and the tests check that end. For a vocabulary that is not a
  power of two, the float32 product can round up to an integer that the exact product
  is just below (3 of 2 million random uniforms at $V = 40$); the statement defines
  the float32 product, and the tests stay away from those values so that a float64
  product is accepted too.
- **Exact comparisons.** Tokens and masks cannot be compared "within tolerance", so
  the tests keep every decision away from rounding, as listed in the statement. One
  thing is therefore not tested: whether a sum exactly equal to the bound is accepted
  ($\le$, as specified and as in the library) or not.

## Common mistakes

- **The position's own entropy in the sum** (`cumsum(ranked) <= bound`). The most
  certain position is then rejected whenever its entropy alone exceeds the bound, and
  a step can accept nothing.
- **Each entropy against the bound,** with no running sum. That is a per-position
  confidence threshold: with a thousand positions of entropy 0.09 it accepts all of
  them at a bound of 0.1, where the rule accepts two.
- **Descending order,** which accepts the least certain positions.
- **Keeping the current token where a position is rejected.** That is how a masked
  sampler behaves. Here the rejected positions are renoised.
- **Entropy of the unscaled logits.** The entropy must be that of the distribution the
  candidates were drawn from; and the reverse, sampling at temperature 1.
- **The schedule reversed.** `n` counts steps remaining: hot first.
- **An unstable sort.** `torch.sort` without `stable=True` is what the library calls,
  and it returns equal keys in index order only for short inputs (with torch 2.15 on
  this machine, 16 elements kept their order in 200 of 200 trials and 32 elements in
  none). The case with 100
  identical rows among 256 positions exists for this: with three bounds that admit 1,
  37, and 99 of them, any order other than by index is caught.
- **Ties to the higher index.**
- **An index offset as the tie-break,** `entropy + index * eps`. It is wrong in two
  different ways, depending on `eps`. Too large, and it reorders entropies that are
  not equal: the tests have two entropies $2.4 \cdot 10^{-5}$ apart, 255 positions
  apart, the smaller one last, so any `eps` above about $10^{-7}$ ranks them
  the wrong way round. Too small, and it disappears: float32 values near 1 are
  $1.2 \cdot 10^{-7}$ apart, so `eps = 3e-8` leaves neighboring equal entropies
  equal, and the order is again whatever the sort does. A small offset *on top of a
  stable sort* changes nothing on inputs that keep distinct entropies $10^{-5}$ apart,
  as the statement promises, and is accepted; it is also unnecessary.
- **Entropy in bits.** The bound is in nats.
- **`round` for the noise token,** or reusing `u_sample` for it.
- **`cdf < threshold`** in the count: with `u = 0` it returns token 0 even when its
  probability is 0.
- **The argmax canvas confused with the sampled canvas,** or updated only where a
  position was accepted.
- **Writing the result into `current`.**
- **Calling `ops.eb_accept`** from the kit, which is this problem's steps 3 and 4.

## Complexity

$O(LV)$ for the softmax, the cumulative sums, and the entropies, and $O(L \log L)$ for
the sort. The denoiser's forward pass dominates by far; what the step buys is fewer of
those passes.

## In the wild

Checked on 2026-11-21 against
`models/diffusion_gemma/generation_diffusion_gemma.py` of `transformers` 5.23.0, by
reading it and by running it, and against the files of
`google/diffusiongemma-26B-A4B-it` on the Hugging Face Hub.

- **`LinearTemperatureScheduleLogitsProcessor`** returns
  `scores / (t_min + (t_max - t_min) * (cur_step / max_denoising_steps))`, and
  `generate` iterates `cur_step` over `reversed(range(1, max_denoising_steps + 1))`.
- **`_denoising_step`** applies the processor, takes the softmax, samples one token per
  position with `torch.multinomial`, takes the argmax of the processed logits, and
  calls the sampler.
- **`EntropyBoundSampler.accept_canvas`** computes
  `Categorical(logits=logits).entropy()`, sorts ascending, and accepts where
  `cumsum(sorted) - sorted <= entropy_bound`; **`renoise_canvas`** replaces every
  position that was not accepted with `torch.randint(0, vocab_size)`.
- **Values.** `generation_config.json`: `max_denoising_steps` 48, `t_min` 0.4, `t_max`
  0.8, `entropy_bound` 0.1 (also the library's defaults). `config.json`:
  `canvas_length` 256, vocabulary 262144.

The public tests compare with `_denoising_step` itself, called with fixed logits
(`fixtures/make.py`). Two of its random draws are replaced there, because a library
generator cannot be matched across frameworks: `torch.multinomial` by the inverse-CDF
rule of step 2 applied to the probabilities the step passes it, and the sampler's
`initialize_canvas` by $\lfloor u V \rfloor$. Everything else in the expected arrays is
the library's arithmetic.

Differences from the shipped code, all of them:

- **Random draws** come from supplied uniforms, as just described. The distributions
  are the same (categorical with the softmax probabilities; uniform over the
  vocabulary).
- **Ties.** `torch.sort` is called without `stable=True`, so the library does not
  promise an order among equal entropies. "Lowest index first" is this problem's rule.
  The MLX reference ranks by comparing every pair instead of sorting, so that it does
  not rest on what `mx.argsort` does with equal keys.
- **The temperature** is a Python float here; the library computes it in float32.
- **One canvas.** The library works on a batch and keeps finished batch items
  unchanged; the stopping rule and the self-conditioning logits it returns belong to
  `block-diffusion-decode`.
- **The entropy bound must be a positive float** in the library's configuration
  class; this problem assumes it.

The sampler's docstring credits "Accelerated Sampling from Masked Diffusion Models via
Entropy Bounded Unmasking" (arXiv 2505.24857). Only its abstract was read for this
problem: it proposes the EB-Sampler for masked diffusion models, which unmasks several
tokens per function evaluation under an error tolerance. Nothing else here is taken
from the paper; the rule above is the library's, applied to uniform noise instead of
masks.
