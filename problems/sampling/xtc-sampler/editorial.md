## The idea

Truncation samplers (top-k, top-p, min-p) cut the tail and keep the head. They make
output more coherent and, for the same reason, more predictable: the model's favorite
continuation wins most of the time, and across a long story the same phrases come
back. XTC attacks that directly. Where the model has **several** acceptable
continuations, it sometimes removes the most obvious ones.

"Acceptable" is defined by the threshold: a token with probability at least
`threshold` is a top choice, something the model would plausibly say. The rule is

> if a row has at least two top choices, remove all of them except the least likely.

Two properties follow, and they are the design:

- **The result is never worse than "acceptable".** The survivor is itself a top
  choice, with probability at least `threshold`, so after XTC the most likely
  remaining token is one the model rated plausible. The sampler does not fall
  through to the tail.
- **Certainty is respected.** With one top choice there is no alternative of
  comparable quality, and nothing happens. A threshold above 0.5 therefore disables
  the sampler: two probabilities above one half do not fit in a distribution.

The coin (`u < probability`) makes the intervention occasional, so the text is a
mixture of ordinary sampling and steps where the obvious choices were unavailable.

## Reference solution (PyTorch)

```python
def xtc_filter(logits, threshold, probability, u):
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    floor = torch.where(above, logits, float("inf")).amin(dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = torch.where(candidate, index, -1).amax(dim=-1, keepdim=True)
    remove = fires & above & (index != survivor)
    return torch.where(remove, float("-inf"), logits)
```

- The softmax is needed only for `above`. The survivor is found among the logits,
  where "least likely" is "smallest" and ties are exact.
- There is no sort. The original implementations walk a sorted candidate list
  because their data structure is one; on a dense tensor, a masked minimum finds
  the last top choice in one pass.
- A row with no top choice has `floor = +inf`, `candidate` all false, and
  `survivor = -1`; it does not matter, since `fires` is false for it.

## Common mistakes

- **Removing every top choice.** The off-by-one of this sampler. The best remaining
  token then has probability below the threshold, possibly far below, and the
  renormalized distribution is made of tokens the model rejected.
- **Removing only the top token.** That is a different, much weaker sampler: with
  probabilities 0.45, 0.25, 0.15 it leaves 0.25 as the favorite, and the next step
  is as predictable as before.
- **Acting on a single top choice.** The token that was certainly right becomes
  impossible. The condition is on the *count* of top choices, not on the top
  probability.
- **Keeping the most likely top choice** and removing the rest. A plausible reading
  of "exclude top choices, keep one" that turns XTC into a truncation filter.
- **An inverted coin** (`u >= probability`, or `u > 1 - probability` when the
  uniforms are not symmetric in your tests). `probability` is the chance of acting.
- **One coin for the batch.** All requests in a batch are then filtered on the same
  steps.
- **Ties.** With equal logits among the least likely top choices, "keep the last
  one" has to say which. The rule here keeps one token; keeping all tied tokens is
  also defensible (see below), but it is a different function, and with two tokens
  at 0.5 each it never acts.
- **`>` for the threshold.** Invisible except at exact equality, which happens with
  uniform rows: two equal tokens at a threshold of 0.5.

## Complexity

$O(B \cdot V)$ time and memory: a softmax, two masked reductions, and a select.

## In the wild

XTC was proposed by Philipp Emanuel Weidmann (p-e-w) as text-generation-webui pull
request 6335 (2024) and spread through local inference engines for two years before
it was written up as a paper, "XTC: Head-Aware Sampling by Excluding Top
Choices" (arXiv 2608.22758, 2026), whose abstract states the rule as here: tokens
above an absolute plausibility threshold are flagged, the rule applies only when at
least two pass, and with a given probability all but the weakest of them are
removed. This problem follows the rule as merged in llama.cpp
(`llama_sample_xtc_apply` in `src/llama-sampler.cpp`): return early when
`probability <= 0`, `threshold > 0.5`, or fewer than two candidates remain; draw a
uniform number and return when it is greater than `probability`; sort the
candidates by probability; find the last position whose probability is
`>= threshold`; and, if that position is not the first, drop every candidate before
it. llama.cpp's defaults are `--xtc-probability 0.0` and `--xtc-threshold 0.1`, and
XTC runs after min-p and before temperature in its default sampler order, so the
threshold applies to the probabilities before temperature scaling.

Where implementations differ:

- **Ties.** llama.cpp keeps exactly one token, whichever its sort placed last among
  equal probabilities; that order is not specified. This problem makes it the
  highest index. mlx-lm's `apply_xtc` instead removes the tokens whose probability
  is strictly greater than the smallest above-threshold probability, so every token
  tied for last survives.
- **The threshold comparison.** llama.cpp uses `p >= threshold`; mlx-lm uses
  `probs > xtc_threshold`.
- **The coin.** llama.cpp skips when `chance > probability` and mlx-lm when
  `uniform > xtc_probability`, so both act when the draw equals the probability;
  here the test is `u < probability`, which makes `probability = 0` exactly "never"
  for uniforms in $[0, 1)$.
- **Guards.** llama.cpp has a `min_keep` argument and will not act if fewer
  candidates than that would remain. mlx-lm takes a list of special tokens (end of
  sequence, newline) that XTC is never allowed to remove, so the sampler cannot
  prevent a generation from ending; llama.cpp has no such list.
- **Temperature.** Because XTC reads probabilities, its effect depends on where it
  sits relative to temperature. Before temperature (llama.cpp's default order), the
  set of top choices is a property of the model's distribution; after a high
  temperature, more tokens clear the threshold and more are removed.
