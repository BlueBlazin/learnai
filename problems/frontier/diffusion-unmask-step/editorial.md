## The idea

A masked-diffusion language model is trained to predict masked tokens given the unmasked ones across all masking ratios. Generation runs that process in reverse by starting with all target positions masked and unmasking them over $T$ steps. In each step, one forward pass produces a token distribution for every masked position independently. Committing all positions at once would sample each token without knowing what its neighbors become. Committing only a subset and running another forward pass lets later positions condition on the tokens settled earlier.

**Choosing which positions to commit.** Draw a candidate token at every position, score each candidate by the model's probability for it, and commit the top $n$. Easy positions such as function words and forced continuations settle first and provide context for harder positions.

**Scoring the drawn candidate.** When `temperature > 0`, sampling can pick a token other than the model's argmax. The confidence score must be the probability of the token that will actually be written, not the maximum probability in that row. Scoring by the row's maximum would commit an unlikely sample with a confidence it does not have. The confidence also uses the unscaled distribution at temperature 1, because temperature shapes the sampling distribution rather than the model's own estimate of token probability.

**Structural constraints.** Two rules are built into the decoding step rather than learned. First, already-unmasked tokens never change, so only masked positions may be selected regardless of how confident the model is elsewhere. Second, the mask token is never a valid prediction, so its logit is set to $-\infty$ before any softmax is computed. That prevents it from being sampled or taking probability mass away from real tokens.

**Splitting the count across steps.** Under a linear noise schedule, the expected number of tokens revealed per step is constant. Dividing `n_masked` evenly across `steps` and adding 1 to the first `n_masked % steps` steps ensures the total equals `n_masked`. Rounding every step down would leave mask tokens in the final output.

## Reference solution (PyTorch)

```python
def schedule(n_masked, steps):
    base, extra = divmod(n_masked, steps)
    return [base + 1 if i < extra else base for i in range(steps)]


def unmask_step(tokens, logits, mask_id, n_unmask, temperature, uniforms):
    masked = tokens == mask_id
    logits = torch.where(torch.arange(logits.shape[-1]) == mask_id, float("-inf"), logits)

    if temperature == 0:
        candidates = torch.argmax(logits, dim=-1)
    else:
        cdf = torch.cumsum(torch.softmax(logits / temperature, dim=-1), dim=-1)
        candidates = (cdf <= uniforms[:, None] * cdf[:, -1:]).sum(dim=-1)

    probs = torch.softmax(logits, dim=-1)
    confidence = torch.take_along_dim(probs, candidates[:, None], dim=-1)[:, 0]
    confidence = torch.where(masked, confidence, float("-inf"))

    n = min(n_unmask, int(masked.sum()))
    order = torch.argsort(confidence, descending=True, stable=True)
    rank = torch.argsort(order)
    return torch.where(rank < n, candidates, tokens)
```

Using `rank < n` avoids indexing with a variable-length slice. That keeps the tensor shapes static for JAX compilation and works in MLX, where boolean-mask indexing is not supported. `torch.argsort` requires `stable=True` to preserve index order on ties. In JAX and MLX, `jnp.argsort` and `mx.argsort` are already stable, so sorting `-confidence` in ascending order produces the same descending rank. Setting the mask token's column to `-inf` gives it probability 0 in both softmaxes, so its cumulative sum repeats the previous entry and it can never be the first index to exceed the threshold.

## Common mistakes

- **Committing unmasked positions.** Without setting unmasked confidences to `-inf`, a confident prediction at an unmasked position can take one of the $n$ slots, overwriting a committed or prompt token and committing fewer masked positions than scheduled.
- **Using the argmax probability as confidence.** `probs.max(-1)` matches the specification only at temperature 0. When sampling, it scores positions by a token they did not draw.
- **Using the tempered distribution for confidence.** Computing confidence from `softmax(logits / temperature)` alters the ranking whenever positions have different entropies.
- **Mishandling temperature during sampling.** Ignoring `temperature` or dividing probabilities instead of logits produces the wrong sampling distribution.
- **Leaving the mask token in the vocabulary.** A model's mask logit is not guaranteed to be small. Leaving it unmasked lets positions commit to `mask_id` and reduces the probability mass on valid tokens.
- **Indexing `uniforms` by masked-position rank.** Using `uniforms[k]` for the $k$-th masked position instead of `uniforms[i]` for position $i$ changes which uniform draw each position receives and fails exact comparison.
- **Not clamping `n` to the masked count.** Checking `rank < n_unmask` when fewer than `n_unmask` positions are masked lets the extra slots spill onto unmasked positions.
- **Using an unstable sort or `topk`.** `torch.topk` and default `torch.argsort` do not guarantee that ties go to the lowest index.
- **Rounding the schedule down or dumping the remainder at the end.** Rounding down leaves $n \bmod T$ masks in the output. Putting the entire remainder into the final step commits the most tokens when the fewest re-prediction steps remain.
- **Drawing random numbers inside `unmask_step`.** The harness forbids framework RNG calls because randomness must come from `uniforms`.

## Complexity

Each step performs one or two softmaxes, a cumulative sum, and a sort over sequence positions, costing $O(LV + L \log L)$ time. That overhead is negligible compared to the transformer forward pass that produces `logits`. Generating $L$ tokens takes $T$ full-sequence forward passes, whereas an autoregressive decoder with a KV cache takes $L$ single-token passes. Masked diffusion trades fewer, wider passes for the loss of a reusable KV cache, since any position's input can change between steps.

## In the wild

What is established in the literature:

- **LLaDA** ("Large Language Diffusion Models", arXiv 2502.09992) describes this reverse process in its inference section and appendix algorithms. While the theoretical reverse process remasks a random fraction $s / t$ of tokens when stepping from time $t$ to $s$, the paper finds that remasking the *lowest-confidence* predictions works better in practice. Its official `generate.py` matches this problem closely. `get_num_transfer_tokens` implements `schedule` (derived from the linear noise schedule), candidates are drawn via Gumbel-max with temperature-scaled logits, confidence is the candidate's entry in `softmax(logits)`, unmasked positions receive confidence $-\infty$, and the top-$k$ positions are transferred. It also offers a `random` strategy that replaces confidences with random numbers.
- **MDLM** ("Simple and Effective Masked Diffusion Language Models", arXiv 2406.07524) formalizes the two structural rules as parameterization properties: zero masking probabilities (setting the mask token's logit to $-\infty$) and carry-over unmasking (copying unmasked tokens unchanged through the reverse process).
- **MaskGIT** ("MaskGIT: Masked Generative Image Transformer", arXiv 2202.04200) introduced this parallel predict-and-refine decoding paradigm for masked image generation.

What this problem simplifies or chooses:

- Candidates are drawn by inverse-CDF sampling with one uniform number per position. LLaDA's code uses Gumbel-max with one random number per logit, which defines the same distribution but yields different concrete samples.
- The mask token is explicitly masked out as in MDLM. LLaDA's published inference code does not mask that logit explicitly.
- This problem covers a single unbatched sequence without classifier-free guidance, semi-autoregressive block generation (used by LLaDA's instruct model to run this step block by block from left to right), or random remasking.
- LLaDA's low-confidence algorithm states a target number of unmasked tokens per timestep. With uniform timesteps, that reduces to the even split used here up to rounding.
