## The idea

A masked-diffusion language model is trained to fill in masked tokens given the
unmasked ones, at every masking ratio. Generation runs that in reverse: start with
everything masked and, over $T$ steps, unmask. In each step the model predicts a
distribution for every masked position in one forward pass. The predictions are
made independently per position given the current sequence, so committing all of
them at once would sample each token without knowing its neighbors. Committing only
some and predicting again lets the later positions condition on the earlier ones.
The step is the rule for "which some".

**Which ones: the most confident.** Draw a candidate everywhere, score each by the
probability the model gives it, commit the top $n$. The effect is that easy
positions (function words, forced continuations) are settled first and become
context for the hard ones.

**Confidence of what.** With a temperature above 0 the candidate is a sample and
need not be the model's favorite. The score must be the probability of the token
that will actually be written. Scoring a position by its favorite's probability
while writing something else commits unlucky draws with a confidence they do not
have. The score uses the model's distribution at temperature 1: the temperature
shapes the draw, not the model's opinion of it.

**What may change.** Two facts about the process are structural, not learned. An
unmasked token stays: only masked positions are ever candidates, however confident
the model is elsewhere. And the mask token is never an output: its logit is removed
before anything else, so it can neither be drawn nor take probability mass.

**How many.** With a linear noise schedule the expected number of tokens revealed
per step is constant, so the schedule is an even split of the masked count, with
the remainder spread over the first steps. Rounding each step down instead leaves
masks in the final output.

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

The commit is written without indexing by a variable-length list: `rank < n` is a
mask of fixed shape, which keeps JAX from compiling a new program for every `n` and
suits MLX, where boolean-mask reads are not available. `torch.argsort` needs
`stable=True`; `jnp.argsort` and `mx.argsort` are stable, so there the descending
order is a sort of the negated confidences. The mask token's column is `-inf` in
both softmaxes, so it has probability exactly 0 and, since its cumulative sum
repeats the previous one, it is never the first index to exceed the threshold.

## Common mistakes

- **Committing unmasked positions.** Without `-inf` confidence at unmasked
  positions, a confident prediction there takes one of the $n$ places and
  overwrites a committed token (or a prompt token). The step then also commits
  fewer masks than scheduled.
- **Confidence of the argmax.** `probs.max(-1)` is the right score only at
  temperature 0. With sampling it ranks positions by a token they are not getting.
- **Confidence from the tempered distribution.** Using
  `softmax(logits / temperature)` for the score changes the ranking whenever
  positions differ in entropy.
- **Ignoring the temperature in the draw**, or applying it to probabilities instead
  of logits.
- **Leaving the mask token in.** A model's mask logit is not trained to be small. If
  it is not removed, positions get "committed" to the mask token and the real
  tokens lose probability.
- **Uniforms indexed by rank.** `uniforms[k]` for the $k$-th masked position instead
  of `uniforms[i]` for position $i$ gives a valid sampler with a different stream;
  the statement pins the stream down so results can be compared.
- **Not limiting `n` to the number of masks.** With `rank < n_unmask` and fewer
  masks than that, the surplus places fall on unmasked positions.
- **Unstable top-k.** `torch.topk` and a default `torch.argsort` do not promise an
  order among equal values; ties must go to the lowest index.
- **A schedule that rounds down**, which leaves $n \bmod T$ masks at the end, or one
  that puts the whole remainder into the last step, which commits the most tokens
  with the least re-prediction.
- **Drawing your own noise.** The harness refuses a framework generator; the
  uniforms are an argument.

## Complexity

One softmax or two, a cumulative sum, and a sort over the positions:
$O(LV + L \log L)$ per step, negligible next to the forward pass that produced the
logits. A generation of $L$ tokens costs $T$ forward passes over the whole
sequence, where an autoregressive decoder with a KV cache costs $L$ single-token
passes; the trade is fewer, wider passes, and no KV cache to reuse, because every
position's input can change between steps.

## In the wild

What is established:

- **LLaDA** ("Large Language Diffusion Models", arXiv 2502.09992) describes the
  reverse process in its section on inference and in two appendix algorithms. The
  plain version predicts all masked tokens and then remasks a fraction $s/t$ of
  them at random when going from time $t$ to $s$; the paper says the remasking
  should in principle be purely random and that, in practice, remasking the
  predictions with the *lowest confidence* works better. Confidence is the
  predicted token's probability, and timesteps are uniform by default. The
  official `generate.py` is this problem's step: `get_num_transfer_tokens` is
  `schedule` (the comment there derives the even split from the linear noise
  schedule), the candidate is an argmax of the logits with Gumbel noise scaled by
  the temperature, the confidence is the candidate's entry of `softmax(logits)`,
  positions that are not masked get confidence $-\infty$, and the top-$k$ are
  transferred. It also has a `random` strategy, which replaces the confidences by
  random numbers.
- **MDLM** ("Simple and Effective Masked Diffusion Language Models", arXiv
  2406.07524) states the two structural rules as properties of its
  parameterization: zero masking probabilities (the mask token's logit is set to
  $-\infty$) and carry-over unmasking (unmasked inputs are copied), and notes that
  in the reverse process unmasked tokens remain unchanged.
- The idea of predicting everything in parallel and refining over a few passes
  comes from masked image generation: the abstract of "MaskGIT: Masked Generative
  Image Transformer" (arXiv 2202.04200) describes generating all tokens at once
  and refining them iteratively.

What this problem simplifies or chooses:

- The candidate is drawn by inverse-CDF sampling from one uniform number per
  position. LLaDA's code uses Gumbel-max with one number per logit; the two give
  the same distribution, not the same tokens.
- The mask token is excluded as in MDLM. The LLaDA code as published does not mask
  that logit explicitly.
- One sequence, no batch; no classifier-free guidance; no semi-autoregressive
  blocks (LLaDA's instruct model generates block by block, left to right, running
  this step inside each block); no random-remasking variant.
- The LLaDA paper's low-confidence algorithm states the count per step as a target
  number of unmasked tokens at each time; with uniform timesteps that is the even
  split used here, up to rounding.
