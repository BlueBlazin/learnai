## The idea

A text-diffusion model predicts every position of a canvas in a single forward pass, but its output is a product of independent per-position distributions. Sampling every position at once from that product ignores dependencies between tokens. For example, two positions that are each "either *New York* or *Los Angeles*" can sample as *New Angeles*. Sampling only one position per pass avoids that error, but loses the speed advantage over autoregressive decoding.

Entropy-bounded acceptance sits between these two extremes. Positions with near-zero entropy are already determined, so sampling them together loses little joint accuracy. The sampler orders positions by increasing entropy and stops once the entropies of the already-accepted positions exceed a budget. Comparing each position against the sum of entropies accepted **before** it has two important effects:

- The first-ranked position is compared against 0 and is always accepted, guaranteeing that every step makes progress even when every position is still uncertain.
- The position that crosses the budget is itself accepted. Equivalently, the rule accepts the $k$ lowest-entropy positions satisfying $\sum_{i \le k} H_i - \max_{i \le k} H_i \le \texttt{entropy\_bound}$, which is the cumulative sum minus its largest (last) term.

How rejected positions are handled separates this sampler from masked diffusion. The canvas starts as uniformly random tokens with no mask token, and every rejected position is **renoised** with a fresh uniform random token rather than keeping its old token. Acceptance is recomputed from scratch on the new logits at every step, so a position accepted on one step can be rejected and renoised on the next. The tokens in `current` influence the new canvas only through the logits the denoiser produced from them, which is why a correct `denoise_step` never reads `current`.

The temperature anneals linearly from $t_\text{max}$ down toward $t_\text{min}$ as the remaining step count `n` decreases. Because temperature scaling happens before both sampling and entropy computation, lower temperatures also reduce entropies and allow more positions to pass the bound. Finally, `argmax_canvas` is the running greedy draft that the library streams, checks for convergence, and appends to the output when the canvas finishes. Dividing by a positive temperature does not change an argmax.

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

`argsort(order)` computes the inverse permutation. Because `order[r]` is the position at rank `r`, `argsort(order)[i]` is the rank of position `i`.

## Numerics

- **Zero probabilities.** At temperature 0.4, a logit gap of 42 becomes 105 after scaling, and `exp(-105)` underflows to 0 in float32. `log_softmax` remains finite at `-105`, so `p * log_p` evaluates cleanly to `0 * (-105) = 0`. Computing `torch.log(p)` instead yields `0 * -inf = NaN`, poisoning the row's entropy and rejecting the most settled positions on the canvas.
- **Noise token range.** Ensuring $\lfloor u V \rfloor < V$ requires $u V < V$ after float32 rounding. For the largest float32 below 1 ($1 - 2^{-24}$), the product $V(1 - 2^{-24})$ rounds below $V$ for all tested vocabularies (every $V$ up to 70,000, plus $2^{18}$ and $2^{24}$), so no clamp is needed. When $V$ is not a power of two, a float32 product can occasionally round up to an integer that the exact product sits just below (3 out of 2 million random uniforms at $V = 40$). The specification defines the product in float32, and the tests stay away from those boundary values so float64 products also match.
- **Exact comparisons.** Discrete tokens and boolean masks cannot be compared with a numeric tolerance, so the tests keep all decision thresholds away from rounding boundaries. One edge case is therefore untested: whether a prefix sum exactly equal to `entropy_bound` is accepted ($\le$, as specified and implemented in the library) or rejected.

## Common mistakes

- **Including the position's own entropy in the sum** (`cumsum(ranked) <= bound`). Whenever the lowest entropy alone exceeds the bound, zero positions are accepted.
- **Comparing each entropy individually against the bound** without a running sum. That implements a per-position threshold. At a bound of 0.1, it would accept a thousand positions of entropy 0.09, whereas the entropy-bounded rule accepts only two.
- **Sorting in descending order**, which accepts the least certain positions first.
- **Keeping `current` at rejected positions** instead of renoising them. That matches masked diffusion, not DiffusionGemma.
- **Computing entropy on unscaled logits** (or sampling at temperature 1). Both sampling and entropy must use `logits / temperature`.
- **Reversing the temperature schedule.** `n` counts remaining steps, so the schedule starts hot at `n = n_max` and cools down.
- **Using an unstable sort.** `torch.sort` without `stable=True` (which the library uses) happens to preserve index order on short arrays but scrambles ties on longer ones (on torch 2.15 on this machine, 16 elements preserved order in 200 of 200 trials, while 32 elements did so in 0 of 200). A test with 100 identical rows among 256 positions and bounds admitting 1, 37, and 99 of them catches any unstable sort.
- **Breaking ties toward the higher index.**
- **Adding a small index offset as a tie-breaker** (`entropy + index * eps`). This fails for both large and small `eps`. If `eps` is too large (above roughly $10^{-7}$), it flips distinct entropies in the test where two entropies $2.4 \cdot 10^{-5}$ apart sit 255 positions apart with the smaller one last. If `eps` is too small (such as `3e-8`), float32 spacing near 1 is $1.2 \cdot 10^{-7}$, so adjacent equal entropies stay equal and fall back to the sort's tie behavior. Adding a tiny offset on top of an already stable sort is unnecessary.
- **Using base-2 entropy (bits).** The bound is in natural log units (nats).
- **Using `round` for the noise token** or reusing `u_sample` instead of `u_noise`.
- **Using strict `<` in `(cdf < threshold).sum(dim=-1)`.** When `u_sample` is 0, that returns token 0 even if its probability is 0.
- **Confusing `argmax_canvas` with `canvas`**, or updating `argmax_canvas` only at accepted positions.
- **Writing results into `current` in place.**
- **Calling `ops.eb_accept`** from `learnai.kit`, which implements steps 3 and 4 and is forbidden.

## Complexity

Softmax, cumulative sums, and entropies take $O(LV)$ time, and sorting takes $O(L \log L)$ time. The denoiser's forward pass dominates runtime, and this step reduces the total number of forward passes needed.

## In the wild

Checked on 2026-11-21 against `models/diffusion_gemma/generation_diffusion_gemma.py` in `transformers` 5.23.0 (both by reading the source and by running it) and against `google/diffusiongemma-26B-A4B-it` on the Hugging Face Hub.

- **`LinearTemperatureScheduleLogitsProcessor`** computes `scores / (t_min + (t_max - t_min) * (cur_step / max_denoising_steps))`, with `generate` iterating `cur_step` over `reversed(range(1, max_denoising_steps + 1))`.
- **`_denoising_step`** applies the logits processor, takes the softmax, samples one token per position via `torch.multinomial`, computes the argmax of the processed logits, and invokes the sampler.
- **`EntropyBoundSampler.accept_canvas`** computes `Categorical(logits=logits).entropy()`, sorts ascending, and accepts positions where `cumsum(sorted) - sorted <= entropy_bound`. **`renoise_canvas`** replaces every rejected position with `torch.randint(0, vocab_size)`.
- **Default hyperparameters.** In `generation_config.json`, `max_denoising_steps` is 48, `t_min` is 0.4, `t_max` is 0.8, and `entropy_bound` is 0.1. In `config.json`, `canvas_length` is 256 and `vocab_size` is 262144.

Public tests compare directly against `_denoising_step` called on fixed logits (`fixtures/make.py`). Because random number generators differ across frameworks, two random draws are replaced in the fixture script: `torch.multinomial` is replaced by the inverse-CDF rule of step 2 on the same probabilities, and `initialize_canvas` is replaced by $\lfloor u V \rfloor$. All other expected values come directly from the library's arithmetic.

Differences from the shipped code:

- **Deterministic randomness.** Uniforms are passed as inputs rather than drawn from an RNG, preserving the same categorical and uniform distributions.
- **Tie-breaking.** The library calls `torch.sort` without `stable=True`, so its tie order is unspecified. This problem specifies lowest index first. The MLX reference ranks positions via pairwise comparison so it does not rely on `mx.argsort` stability.
- **Temperature dtype.** `step_temperature` returns a Python `float`, whereas the library computes it in float32.
- **Single canvas.** The library operates on a batch and leaves finished sequences unchanged. The stopping rule and self-conditioning logits appear in `block-diffusion-decode`.
- **Positive entropy bound.** The library's config requires a positive float, which this problem assumes.

The sampler's docstring credits *Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking* (arXiv 2505.24857). Only its abstract was consulted for this problem. That paper introduces the EB-Sampler for unmasking multiple tokens per step under an error tolerance, while the rule implemented here is the library's adaptation to uniform renoising instead of masks.
