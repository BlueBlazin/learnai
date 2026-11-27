## The idea

Truncation samplers like top-k, top-p, and min-p cut off the tail of the distribution and keep the head. That improves coherence, but it also makes long generations repetitive because the model's favorite continuations win most of the time. XTC takes the opposite approach. When the model has **several** plausible continuations, XTC occasionally removes the most obvious ones.

A token is considered plausible (a **top choice**) when its probability is at least `threshold`. The core rule is:

> If a row has at least two top choices, remove all of them except the least likely one.

Two safety properties follow directly from this rule:

- **The best remaining token is still plausible.** Because the surviving top choice has probability at least `threshold`, the sampler never falls through to the low-probability tail.
- **High-confidence steps stay untouched.** When only one token reaches `threshold`, there is no comparable alternative, so the row is left alone. Consequently, any `threshold > 0.5` disables XTC because two probabilities cannot both exceed 0.5.

The per-row coin flip (`u < probability`) makes this intervention occasional, mixing normal sampling steps with steps where the most predictable continuations are masked out.

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

- Softmax is needed only to compute `above`. Finding the survivor directly on `logits` makes "least likely" equivalent to "smallest" and ensures exact tie comparisons.
- No sorting is required. Original C++ implementations scan a sorted candidate list because their sampler API already stores candidates that way, whereas on a dense tensor a masked minimum finds the smallest top-choice logit in one pass.
- If a row has no top choices, `floor` is `+inf`, `candidate` is all false, and `survivor` is `-1`. That value is harmless because `fires` is false for that row.

## Common mistakes

- **Removing every top choice.** Failing to spare the least likely top choice leaves only tokens below `threshold`, so the renormalized distribution consists entirely of tokens the model considered unlikely.
- **Removing only the single most likely token.** With probabilities `0.45, 0.25, 0.15`, dropping only `0.45` still leaves `0.25` as the dominant choice and produces a much weaker effect than XTC.
- **Acting on a single top choice.** Checking whether the maximum probability exceeds `threshold` instead of counting top choices removes high-confidence tokens that have no good substitute.
- **Keeping the most likely top choice.** Inverting the survivor rule turns XTC into a standard truncation filter that keeps only the top token.
- **Inverting the coin flip.** Using `u >= probability` (or `u > 1 - probability` on asymmetric test inputs) applies the wrong activation rate, since `probability` is the chance of acting.
- **Sharing one coin flip across the batch.** Drawing a single boolean for all rows forces every sequence in a batch to trigger on the same decoding steps.
- **Mishandling ties.** When multiple top choices share the smallest logit, keeping all of them instead of the highest index implements a different rule and prevents XTC from acting when two tokens tie at `0.5` each.
- **Using `>` instead of `>=` for `threshold`.** This bug stays hidden until an exact boundary case occurs, such as two equal tokens at `threshold = 0.5`.

## Complexity

Time and memory are $O(B \cdot V)$, where $B$ is the batch size and $V$ is the vocabulary size. The implementation runs one softmax, two masked reductions along the vocabulary axis, and a final `where` select.

## In the wild

Philipp Emanuel Weidmann (p-e-w) proposed XTC in text-generation-webui pull request 6335 (2024). It spread across local inference engines for two years before appearing in the paper "XTC: Head-Aware Sampling by Excluding Top Choices" (arXiv 2608.22758, 2026), which states the same rule: flag tokens above a plausibility threshold, act only when at least two pass, and with a given probability remove all but the weakest of them.

This problem matches the rule in llama.cpp (`llama_sample_xtc_apply` in `src/llama-sampler.cpp`). That implementation returns early when `probability <= 0`, `threshold > 0.5`, or fewer than two candidates remain. Otherwise it draws a uniform number, skips if the draw exceeds `probability`, sorts candidates by probability, finds the last candidate with probability `>= threshold`, and drops every candidate before it when that position is not index 0. Its defaults are `--xtc-probability 0.0` and `--xtc-threshold 0.1`, and it runs after min-p and before temperature so the threshold applies to pre-temperature probabilities.

Implementations differ in several details:

- **Ties.** llama.cpp keeps whichever tied token its sort happens to place last, whereas this problem fixes the tie-breaker to the highest index. mlx-lm's `apply_xtc` removes only tokens whose probability is strictly greater than the minimum above-threshold probability, so all tokens tied for last survive.
- **Threshold comparison.** llama.cpp uses `p >= threshold`, while mlx-lm uses `probs > xtc_threshold`.
- **Coin flip.** llama.cpp skips when `chance > probability` and mlx-lm skips when `uniform > xtc_probability`, so both act on exact equality. Using `u < probability` ensures that `probability = 0` never acts for draws in $[0, 1)$.
- **Extra guards.** llama.cpp accepts a `min_keep` parameter and skips filtering if fewer candidates would remain. mlx-lm accepts a list of special tokens (such as EOS and newline) that XTC cannot remove, whereas llama.cpp does not.
- **Sampler order relative to temperature.** Because XTC inspects probabilities, running it before temperature (llama.cpp's default) operates on the model's raw distribution. Running it after a high temperature flattens the distribution, allowing more tokens to clear `threshold` and be removed.
