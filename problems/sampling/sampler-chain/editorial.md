## The idea

A sampler chain is function composition, and its stages do not commute. Each filter
makes its decision from the *current* distribution:

- top-k looks at the order of the logits;
- top-p looks at cumulative probability mass;
- min-p looks at probability ratios to the best token.

Anything that changes those quantities changes what a later filter keeps.

**Penalties first.** A penalty is a correction to the model's opinion ("this token
was just used, make it less likely"). It has to be in place before any filter ranks
the tokens. Otherwise a token that was repeated five times still holds its slot in
the top-k and its share of the nucleus, the filters cut tokens that should have
moved up, and the penalty arrives too late to bring them back.

**Top-k, then top-p, then min-p**, each on the renormalized survivors of the one
before. Top-k is the cheap, coarse cut (in a real engine it is also what makes the
later sorts affordable). Top-p then measures its mass on the survivors: after
`top_k = 3` leaves probabilities 0.4, 0.33, 0.27, a `top_p` of 0.7 keeps two tokens,
although the same three tokens held only 0.3, 0.25, 0.2 of the original
distribution. Min-p is a ratio to the largest probability, so renormalizing does not
change it; it removes what is left of the tail.

**Temperature last.** Dividing logits by $T$ is monotone, so it never changes the
ranking, and top-k does not care where it runs. Top-p and min-p do. With
$\pi_i \propto e^{x_i}$, the tempered distribution is $\pi_i^{1/T}$ renormalized:
for $T > 1$ it is flatter, more tokens are needed to reach mass $p$, and ratios to
the top token rise toward 1; for $T < 1$ the opposite. Applying the temperature
after the filters separates the two decisions: the filters decide *which* tokens are
acceptable, from the model's own distribution, and the temperature decides only
*how evenly* to choose among them. That is why llama.cpp can run with temperatures
well above 1 without sampling the tail.

Because every stage hands on logits with `-inf` for removed tokens, "renormalize the
survivors" never has to be written: the next stage's softmax does it.

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

A masked logit survives all of this as `-inf`: `-inf * penalty`, `-inf - c`, and
`-inf / T` are all `-inf`, and the filters never revive an entry. The count is a
`(B, T, V)` comparison; for long histories a scatter-add into a `(B, V)` buffer does
the same in $O(BT)$.

## Common mistakes

- **Temperature first.** The order of Hugging Face transformers and vLLM, and the
  natural one to write ("scale, then filter"). With $T = 2$ in the worked example it
  keeps three tokens instead of two. Neither order is wrong in the abstract; this
  problem, like llama.cpp, specifies temperature last, and the two are not
  interchangeable.
- **Temperature in the middle.** Moving only min-p behind the temperature looks
  harmless because min-p is "just a ratio", but ratios are what the temperature
  changes: $(\pi_i / \pi_{\max})^{1/T}$.
- **Top-p on the unrenormalized survivors.** Computing the nucleus from the original
  probabilities (or computing all masks from the original logits and intersecting
  them) makes top-p see less mass than there is after top-k, so it keeps too many
  tokens. Chaining through logits avoids it.
- **Penalties after the filters.** The penalized token keeps its slot; the token it
  should have made room for is already `-inf`.
- **Dividing negative logits by the repetition penalty.** The formula in the CTRL
  paper divides always, which makes a repeated token with a negative logit *more*
  likely. llama.cpp's comment on this line calls it "obviously wrong"; multiply when
  the logit is not positive.
- **Presence penalty per occurrence.** That is the frequency penalty. Presence is
  charged once per distinct token.
- **Calling top-p with `p = 1.0`.** In exact arithmetic it keeps everything. In
  float32 the cumulative mass reaches exactly 1.0 before the last tokens of a peaked
  distribution, and "mass before < 1.0" fails for them, so a stage that is switched
  off trims the tail. Engines skip the stage (`if (p >= 1.0f) return;`).
- **Sampling before the temperature.** Computing the probabilities from the filtered
  logits and applying the temperature only to what is returned leaves the draw at
  temperature 1.
- **Drawing your own random numbers.** `torch.multinomial` gives a valid sample that
  no test can reproduce; the uniform is an argument.

## Complexity

The filters sort each row: $O(B\,V \log V)$ time and $O(BV)$ memory. The history
count is $O(BTV)$ as written. Engines avoid sorting the whole vocabulary: llama.cpp
partially sorts only as many candidates as top-k or the nucleus needs, and vLLM
fuses top-k and top-p into one kernel.

## In the wild

The order differs between engines (sources read in November 2026):

| engine | order |
|---|---|
| llama.cpp (`common/common.h`, default `samplers`) | penalties, DRY, top-n-sigma, **top-k**, typical-p, **top-p**, **min-p**, XTC, **temperature**, then the draw |
| mlx-lm (`sample_utils.make_sampler`) | logits processors (logit bias, repetition, presence, frequency penalties), then **top-p**, **min-p**, XTC, **top-k**, and the temperature inside the final categorical draw |
| vLLM (`v1/worker/gpu/sample/sampler.py`) | logits processors (bias, penalties), **temperature**, **min-p**, then **top-k** and **top-p** |
| Hugging Face transformers (`generation/utils.py`) | repetition penalty and other processors, then **temperature**, **top-k**, **top-p**, **min-p** |

So llama.cpp and mlx-lm filter the model's own distribution and apply the
temperature last, while vLLM and transformers temper first. mlx-lm also runs top-k
*after* top-p and min-p, which means its nucleus is measured on the full
distribution, not on the top-k survivors: the "unrenormalized" variant above is a
real engine's behavior. In llama.cpp the order is a parameter (`--samplers`), and a
temperature of 0 replaces the draw by an argmax.

The penalties differ too. This problem uses llama.cpp's formula
(`llama_sampler_penalties_apply`), which counts only the last `penalty_last_n`
tokens (64 by default) and is off by default (`penalty_repeat = 1.0`). The OpenAI
definition of frequency and presence penalties counts generated tokens only, and
transformers' `repetition_penalty` also covers the prompt.

The practical consequence: sampling parameters do not transfer between engines.
"Temperature 1.2, top-p 0.9" tuned on one stack is a different distribution on
another.
