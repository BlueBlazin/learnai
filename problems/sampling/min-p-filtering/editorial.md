## The idea

Top-k and top-p enforce a fixed budget regardless of distribution shape, always
keeping a set number of tokens or a set fraction of total probability mass. Min-p
instead tests each token relative to the best candidate in the same row: is this
token at least `min_p` times as likely as the top token? When one token has
probability 0.9, only tokens above $0.9 \cdot \text{min\_p}$ survive, usually a
handful. When the best token has probability 0.05, the bar drops to
$0.05 \cdot \text{min\_p}$ and hundreds may pass. The filter tightens when the model
is confident and relaxes when it is uncertain.

Let $x_i$ denote the logits of a row, $x_{\max} = \max_j x_j$ the row's largest
logit, and $\pi = \operatorname{softmax}(x)$ the resulting probabilities. Because
the filter compares a ratio of two probabilities from the same row, the softmax
normalizer cancels:

$$
\frac{\pi_i}{\pi_{\max}} = e^{x_i - x_{\max}} \ge \text{min\_p}
\iff x_i \ge x_{\max} + \ln(\text{min\_p}).
$$

In logit space, min-p keeps a fixed-width band of size $-\ln(\text{min\_p})$ below
the top logit (about 3.0 for `min_p = 0.05`). Because shifting a row by a constant,
renormalizing, or masking other tokens never changes logit differences, none of
those operations change which tokens survive. This form also shows why the top token
can never be lost, since its ratio is 1 and $\text{min\_p} \le 1$.

## Reference solution (PyTorch)

```python
def min_p_filter(logits, min_p):
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * torch.amax(probs, dim=-1, keepdim=True)
    return torch.where(probs >= threshold, logits, float("-inf"))
```

- `keepdim=True` gives the row maximum shape `(B, 1)`, producing one cutoff per row.
- Using `>=` handles both boundary values without special cases. When `min_p = 1`,
  the top token equals the cutoff and passes. When `min_p = 0`, the cutoff is 0 and
  every probability passes, including 0 for already-masked tokens whose logits copy
  through as `-inf`.
- The mask is computed from probabilities and applied to `logits`. Setting removed
  entries to `-inf` lets the next stage's softmax assign them probability 0 and
  renormalize the remaining tokens automatically.

The equivalent logit-space form avoids computing softmax altogether:

```python
def min_p_filter(logits, min_p):
    if min_p <= 0:
        return logits
    cutoff = torch.amax(logits, dim=-1, keepdim=True) + math.log(min_p)
    return torch.where(logits >= cutoff, logits, float("-inf"))
```

## Common mistakes

- **Using an absolute cutoff (`probs >= min_p`).** This implements a different
  sampler (sometimes called epsilon sampling). On a confident row it looks similar
  to min-p and can slip past casual testing, but on a flat row over a large
  vocabulary every probability falls below `min_p` and the entire row is masked out.
- **Applying the ratio directly to logits (`logits >= min_p * logits.max()`).**
  Logits are log-probabilities up to an additive constant, so a ratio of
  probabilities corresponds to a *difference* of logits. Scaling logits breaks shift
  invariance, and when the maximum logit is negative the cutoff ends up above the
  maximum logit itself.
- **Using `>` instead of `>=`.** This goes unnoticed on most inputs, but when
  `min_p = 1` it masks out the top token and leaves an empty row.
- **Sharing a single maximum across the batch (`probs.max()` with no axis).** A
  confident row then sets a high cutoff for any flat row batched alongside it.
- **Zeroing out removed entries.** Writing `0` where the mask is false works for
  probabilities, not logits. In logit space, `0` is an ordinary score, so "removed"
  tokens can become some of the most likely candidates.
- **Filtering after temperature when the pipeline expects it before.** Although not
  part of this function, sampler order matters because temperature scales
  $x_i - x_{\max}$, changing which tokens fall inside the band.

## Complexity

$O(B \cdot V)$ time and memory for one softmax (or one row-wise maximum), one
comparison, and one masked select. No sort is needed, which makes min-p cheaper than
top-p.

## In the wild

Min-p was introduced by Nguyen et al., "Turning Up the Heat: Min-p Sampling for
Creative and Coherent LLM Outputs" (arXiv 2407.01082, 2024; ICLR 2025), using this
exact rule of scaling `min_p` by the top token's probability.

`llama.cpp` implements the logit-space form (`min_logit = max_logit + logf(p)`,
keeping `logit >= min_logit`), skips the filter when `p <= 0`, and enables it by
default with `min_p = 0.05` after top-k and top-p and before temperature. `mlx-lm`'s
`apply_min_p` operates on log-probabilities in the same way (removing tokens where
`logprobs < top_logprobs + log(min_p)`) and is disabled by default (`min_p = 0.0`).
Hugging Face's `MinPLogitsWarper` uses the probability form shown in the reference.
All three include a `min_keep` or `min_tokens_to_keep` floor, which changes nothing
at its default of 1 because the top token always passes.
