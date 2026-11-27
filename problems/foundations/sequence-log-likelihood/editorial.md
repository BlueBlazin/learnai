## The idea

An autoregressive language model factorizes the log-probability of a sequence into a sum of next-token log-probabilities:

$$
\log p(x_1, \dots, x_{n-1} \mid x_0) = \sum_{t=0}^{n-2} \log p(x_{t+1} \mid x_0, \dots, x_t) .
$$

At position $t$, the model has read $x_0, \dots, x_t$ and predicts $x_{t+1}$. The logits tensor and the token tensor are therefore offset by one position. Two slices align them into `(B, T - 1)` pairs:

```
logits[:, :-1]   positions 0 .. T-2   the contexts
tokens[:, 1:]    positions 1 .. T-1   the targets
```

**Log-probabilities.** Apply `log_softmax` over the vocabulary axis of the context logits, then gather the single entry indexed by each target token. Computing `log(softmax(x))` instead fails on unlikely targets whose probability underflows float32 to `0.0`, turning a finite log-probability such as $-20000$ into `-inf`.

**Which pairs count.** A pair spans two adjacent positions, and both must be real tokens. With right padding, `mask[:, 1:]` drops the pair from the last real token to the first padding token. With left padding, `mask[:, :-1]` drops the pair from the last padding token to the first real token. Combining both with `mask[:, :-1] & mask[:, 1:]` handles either padding side in one expression.

**Dropping uncounted pairs.** Zero out uncounted positions with `where` rather than multiplying by the mask. Padding tokens are often masked with a `-inf` logit, which gives an uncounted pair a log-probability of `-inf`. Multiplying `-inf` by `0` produces `NaN`.

**Perplexity.** Perplexity is $\exp(-\text{logprob} / n)$, the geometric mean of $1 / p$ across the $n$ counted pairs. If a model assigns uniform probability $1 / k$ to $k$ tokens at every step, its perplexity is exactly $k$. That check confirms that the division by $n$ belongs inside the exponential.

## Reference solution (PyTorch)

```python
def _pair_logprobs(logits, tokens, mask):
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = tokens[:, 1:]
    picked = torch.gather(logp, 2, targets[:, :, None])[:, :, 0]
    counted = mask[:, :-1] & mask[:, 1:]
    return torch.where(counted, picked, torch.zeros_like(picked)), counted


def sequence_logprob(logits, tokens, mask):
    picked, _ = _pair_logprobs(logits, tokens, mask)
    return picked.sum(dim=1)


def perplexity(logits, tokens, mask):
    picked, counted = _pair_logprobs(logits, tokens, mask)
    return torch.exp(-picked.sum(dim=1) / counted.sum(dim=1))
```

JAX uses `jax.nn.log_softmax` and `jnp.take_along_axis`. MLX subtracts `mx.logsumexp` and uses `mx.take_along_axis`. In PyTorch, `-F.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none")` computes the same `picked` tensor in a single call.

## Common mistakes

- **No shift.** Gathering `tokens[t]` from `logits[t]` scores each token under a distribution computed after reading it, producing invalid scores without raising an error.
- **Shifting the wrong way.** Pairing `logits[:, 1:]` with `tokens[:, :-1]` shifts every prediction two positions off.
- **Using `log(softmax(...))`.** Unlikely targets underflow to zero in `softmax`, turning their log-probability and the entire sequence sum into `-inf`.
- **Ignoring the mask or checking only one side.** Using `mask[:, :-1]` alone includes the prediction of the first padding token in right-padded rows. Using `mask[:, 1:]` alone includes the prediction of the first real token from padding in left-padded rows. Each mistake passes tests padded on the opposite side.
- **Multiplying by the mask.** Any uncounted pair with log-probability `-inf` turns into `0 * -inf = NaN`.
- **Dividing by the wrong count.** Averaging over all `T - 1` positions divides short rows by too large a number, artificially lowering their perplexity. Dividing by the number of real tokens is also off by one because $n$ real tokens form $n - 1$ pairs.
- **Taking `exp` of the unnormalized sum.** Computing $\exp(-\text{logprob})$ yields $1 / p(\text{sequence})$, which grows exponentially with sequence length and overflows on real text. Perplexity is always per token.

## Complexity

$O(B\,T\,V)$ time and memory for `log_softmax`, plus $O(B\,T)$ for the gather and reductions.

## In the wild

Training and evaluation code typically writes the shift as `shift_logits = logits[..., :-1, :]` and `shift_labels = labels[..., 1:]`, followed by cross-entropy with an ignore index for padding tokens. Published benchmarks usually report **corpus-level** perplexity, which sums negative log-probabilities across the entire dataset and divides by the total number of scored tokens before taking the exponential. Because rows have different lengths and `exp` is nonlinear, corpus-level perplexity does not equal the average of per-row perplexities. Aggregate with `sequence_logprob` and the per-row pair counts instead.

Serving engines use the same gather. APIs that return `logprobs` read each sampled token's log-probability from the logits one position earlier. Multiple-choice scoring applies `sequence_logprob` over only the continuation tokens. Speculative decoding runs the target model once over a drafted sequence and reads the target probability of each drafted token `t + 1` from the logits at position `t`.
