## The idea

A sampler chain is function composition, and its stages do not commute. Each filter
decides what to keep from the *current* distribution:

- **top-k** looks at the rank order of the logits.
- **top-p** looks at cumulative probability mass.
- **min-p** looks at probability ratios relative to the top token.

Anything that changes those quantities changes what later filters keep.

**Penalties first.** A penalty adjusts the model's raw scores to discourage recently
used tokens, so it must run before any filter ranks them. If penalties ran after the
filters, a heavily repeated token would still occupy a top-k slot and a share of the
top-p nucleus. Other tokens that should have moved up would already be removed, and the
penalty would arrive too late to bring them back.

**Top-k, then top-p, then min-p.** Each filter runs on the renormalized survivors of the
previous one. Top-k makes a cheap coarse cut (and in real engines, it keeps later sorts
affordable). Top-p then measures cumulative mass over the surviving tokens. For example,
if `top_k = 3` leaves renormalized probabilities `0.4`, `0.33`, and `0.27`, then
`top_p = 0.7` keeps two tokens, even though those same three tokens might have held only
`0.3`, `0.25`, and `0.2` of the original distribution. Min-p compares each probability
as a ratio to the largest probability, which is invariant to renormalization, and trims
the remaining tail.

**Temperature last.** Dividing logits by the temperature $\tau$ is strictly monotone, so
it preserves the ranking and leaves top-k unchanged. Top-p and min-p do change. When
$\pi_i \propto e^{x_i}$, scaling logits by $1/\tau$ produces the tempered distribution
$\pi_i^{1/\tau}$ (renormalized). For $\tau > 1$, the distribution flattens, more tokens
are needed to reach mass `top_p`, and ratios to the top token move closer to `1`. For
$\tau < 1$, the opposite happens. Applying temperature after the filters separates two
decisions. The filters choose *which* tokens are acceptable under the model's own
distribution, while temperature controls *how evenly* to sample among them. That is why
llama.cpp can use temperatures well above `1` without pulling in the low-probability
tail.

Because every stage passes logits with `-inf` for removed tokens, renormalization
happens automatically whenever the next stage calls `softmax`.

## Reference solution (PyTorch)

```python
def chain_logits(logits, cfg, history):
    vocab = torch.arange(logits.shape[-1])
    counts = (history[:, :, None] == vocab).sum(dim=1)
    seen = counts > 0
    penalty = cfg["repetition_penalty"]
    x = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)
    x = x - counts * cfg["frequency_penalty"] - seen * cfg["presence_penalty"]

    x = ops.top_k_filter(x, cfg["top_k"])
    if cfg["top_p"] < 1.0:
        x = ops.top_p_filter(x, cfg["top_p"])
    x = ops.min_p_filter(x, cfg["min_p"])
    return x / cfg["temperature"]


def sample_chain(logits, cfg, history, u):
    probs = ops.softmax(chain_logits(logits, cfg, history))
    cdf = torch.cumsum(probs, dim=-1)
    return (cdf <= u[:, None] * cdf[:, -1:]).sum(dim=-1)
```

Masked logits remain `-inf` through every stage because `-inf * penalty`, `-inf - c`,
and `-inf / temperature` all evaluate to `-inf`, and the filters never revive a `-inf`
entry. The history count above uses a `(B, T, V)` comparison. For longer histories, a
scatter-add into a `(B, V)` buffer computes the same counts in $O(B(T + V))$ time.

## Common mistakes

- **Applying temperature first.** This is the order used by Hugging Face transformers
  and vLLM, and it is natural to write "scale, then filter." With `temperature = 2.0` in
  the worked example, it keeps three tokens instead of two. Neither order is wrong in
  general, but this problem specifies llama.cpp's temperature-last order, and the two are
  not equivalent.
- **Applying temperature in the middle.** Placing min-p after temperature looks harmless
  because min-p uses ratios, but temperature changes every ratio to
  $(\pi_i / \pi_{\max})^{1/\tau}$.
- **Running top-p on unrenormalized survivors.** Computing the nucleus from the original
  probabilities (or computing all masks from the original logits and intersecting them)
  makes top-p see less mass than remains after top-k, so it keeps too many tokens.
  Chaining through `-inf`-masked logits avoids this bug.
- **Applying penalties after the filters.** The penalized token keeps its slot, while
  the tokens it should have made room for are already `-inf`.
- **Dividing negative logits by the repetition penalty.** The original CTRL formula
  always divides, which makes a repeated token with a negative logit *more* likely.
  Multiply by the penalty whenever the logit is non-positive.
- **Charging presence penalty per occurrence.** Scaling by the count turns presence
  penalty into a second frequency penalty. Presence is charged once per distinct token.
- **Calling `top_p_filter` when `top_p == 1.0`.** In exact arithmetic this keeps every
  token, but in float32 the cumulative sum of a peaked distribution reaches `1.0` before
  the last tokens. The test "mass before < 1.0" then fails for those tail tokens even
  though the stage is switched off. Skip the filter whenever `top_p >= 1.0`.
- **Sampling before applying temperature.** Computing probabilities from the filtered
  logits and dividing by temperature only for the return value leaves `sample_chain`
  drawing at temperature `1.0`.
- **Drawing your own random numbers.** Calling `torch.multinomial` or similar produces a
  valid sample that tests cannot reproduce. Always use the provided uniform input `u`.

## Complexity

The filters sort each row in $O(B\,V \log V)$ time and $O(BV)$ memory. The history count
takes $O(BTV)$ time as written (or $O(B(T + V))$ with scatter-add). Production engines
avoid sorting the entire vocabulary. llama.cpp partially sorts only as many candidates
as top-k or the nucleus requires, and vLLM fuses top-k and top-p into a single kernel.

## In the wild

Sampling stage order differs across engines (sources read in November 2026):

| engine | order |
|---|---|
| llama.cpp (`common/common.h`, default `samplers`) | penalties, DRY, top-n-sigma, **top-k**, typical-p, **top-p**, **min-p**, XTC, **temperature**, then the draw |
| mlx-lm (`sample_utils.make_sampler`) | logits processors (logit bias, repetition, presence, frequency penalties), then **top-p**, **min-p**, XTC, **top-k**, and the temperature inside the final categorical draw |
| vLLM (`v1/worker/gpu/sample/sampler.py`) | logits processors (bias, penalties), **temperature**, **min-p**, then **top-k** and **top-p** |
| Hugging Face transformers (`generation/utils.py`) | repetition penalty and other processors, then **temperature**, **top-k**, **top-p**, **min-p** |

llama.cpp and mlx-lm filter the model's unscaled distribution and apply temperature
last, whereas vLLM and transformers apply temperature first. mlx-lm also runs top-k
*after* top-p and min-p, measuring its nucleus on the full distribution rather than on
the top-k survivors. In llama.cpp, the order is configurable via `--samplers`, and a
temperature of `0` replaces sampling with argmax.

Penalty definitions also vary. This problem uses llama.cpp's formula
(`llama_sampler_penalties_apply`), which in llama.cpp counts only the last
`penalty_last_n` tokens (64 by default) and is off by default (`penalty_repeat = 1.0`).
OpenAI's frequency and presence penalties count generated tokens only, while
transformers' `repetition_penalty` also includes the prompt.

Because of these differences, sampling parameters do not transfer cleanly across
engines. A setting like `temperature=1.2, top_p=0.9` tuned on one stack defines a
different distribution on another.
