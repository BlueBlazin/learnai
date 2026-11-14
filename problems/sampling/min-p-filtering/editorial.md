## The idea

Top-k and top-p use a budget that ignores the shape of the distribution: always 40
tokens, always 95% of the mass. Min-p asks a question about each token instead: is it
within a factor `min_p` of the best candidate? When one token has probability 0.9,
only tokens above $0.9 \cdot \text{min\_p}$ survive, usually a handful. When the best
token has 0.05, the bar drops to $0.05 \cdot \text{min\_p}$ and hundreds may pass. The
filter is tight when the model is confident and loose when it is not.

Because the test is a ratio of two probabilities of the same row, the softmax
normalizer cancels:

$$
\frac{\pi_i}{\pi_{\max}} = e^{x_i - x_{\max}} \ge \text{min\_p}
\iff x_i \ge x_{\max} + \ln(\text{min\_p}).
$$

So min-p is a fixed-width band below the top logit, of width $-\ln(\text{min\_p})$
(about 3.0 for `min_p = 0.05`). That is why renormalizing, shifting a row by a
constant, or masking other tokens never changes who survives: none of them changes a
difference of logits. It also shows why the top token cannot be lost: its ratio is 1,
and $\text{min\_p} \le 1$.

## Reference solution (PyTorch)

```python
def min_p_filter(logits, min_p):
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * torch.amax(probs, dim=-1, keepdim=True)
    return torch.where(probs >= threshold, logits, float("-inf"))
```

- `keepdim=True` leaves the row maximum with shape `(B, 1)`, one cutoff per row.
- `>=` handles both ends without a special case. For `min_p = 1` the top token equals
  the cutoff and passes. For `min_p = 0` the cutoff is 0 and every probability passes,
  including the 0 of a masked token, whose logit is copied as `-inf`.
- The mask comes from probabilities and is applied to logits. Removed tokens become
  `-inf`, so the next stage's softmax gives them probability 0 and renormalizes the
  rest by itself.

The logit-space form avoids the softmax entirely:

```python
def min_p_filter(logits, min_p):
    if min_p <= 0:
        return logits
    cutoff = torch.amax(logits, dim=-1, keepdim=True) + math.log(min_p)
    return torch.where(logits >= cutoff, logits, float("-inf"))
```

## Common mistakes

- **An absolute cutoff**, `probs >= min_p`. This is a different sampler (sometimes
  called epsilon sampling). For a confident row it behaves much like min-p, which is
  why it survives casual testing; for a flat row over a large vocabulary every
  probability is below the cutoff and the row is emptied.
- **The ratio applied to logits**, `logits >= min_p * logits.max()`. Logits are
  log-probabilities up to a constant, so a ratio of probabilities is a *difference* of
  logits. The scaled version depends on where zero happens to be: shift the row by a
  constant and the survivors change, and for a negative top logit the cutoff is above
  the top logit itself.
- **`>` instead of `>=`.** Invisible for most inputs, and with `min_p = 1` it removes
  the top token and leaves nothing.
- **A top probability shared by the batch**, `probs.max()` with no axis. A confident
  row sets the bar for a flat row batched with it.
- **Zeroing removed entries.** Writing 0 where the mask is false is right for a
  probability vector and wrong for logits, where 0 is an ordinary value: the "removed"
  tokens can end up among the most likely ones.
- **Filtering after temperature when the chain says before.** Not part of this
  function, but the reason order matters in a sampler chain: temperature changes
  $x_i - x_{\max}$, so the band catches different tokens depending on which runs first.

## Complexity

$O(B \cdot V)$ time and memory: one softmax (or one max), one comparison, one select.
No sort is needed, which makes min-p cheaper than top-p.

## In the wild

Min-p was introduced by Nguyen et al., "Turning Up the Heat: Min-p Sampling for
Creative and Coherent LLM Outputs" (arXiv 2407.01082, 2024; ICLR 2025), with exactly
this rule: the base value is scaled by the top token's probability to give the
truncation threshold. llama.cpp implements the logit-space form
(`min_logit = max_logit + logf(p)`, keep `logit >= min_logit`), skips the sampler for
`p <= 0`, and enables it by default with `min_p = 0.05`, after top-k and top-p and
before temperature in its default chain. mlx-lm's `apply_min_p` works on
log-probabilities the same way (`logprobs < top_logprobs + log(min_p)` is removed) and
is off by default (`min_p = 0.0`). Hugging Face's `MinPLogitsWarper` uses the
probability form of the reference. All three add a `min_keep` /
`min_tokens_to_keep` floor; with the default of 1 it changes nothing, since the top
token always passes.
