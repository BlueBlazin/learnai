## The idea

An autoregressive model factorizes the probability of a sequence into next-token
predictions:

$$
\log p(x_1, \dots, x_{n-1} \mid x_0) = \sum_{t=0}^{n-2} \log p(x_{t+1} \mid x_0, \dots, x_t) .
$$

The model's output at position $t$ is the distribution on the right-hand side: it was
computed from $x_0, \dots, x_t$ and it is about $x_{t+1}$. So the tensor of logits and
the tensor of tokens are offset by one, and lining them up is a pair of slices:

```
logits[:, :-1]   positions 0 .. T-2   the contexts
tokens[:, 1:]    positions 1 .. T-1   the targets
```

Everything after that is bookkeeping on `(B, T - 1)` tensors.

**Log-probabilities.** `log_softmax` of the context logits, then one entry per
position: the one indexed by the target. This is a gather along the vocabulary axis.
It must be `log_softmax`: a target that the model considers impossible has a
probability far below the smallest float32, and `log(softmax(x))` returns `-inf` where
the true log-probability is an ordinary number such as $-20000$.

**Which pairs count.** A pair consists of two positions, and both must be real. With
right padding the binding condition is the target (`mask[:, 1:]`): the pair made of
the last real token and the first padding token must go. With left padding it is the
context (`mask[:, :-1]`): the first real token would otherwise be scored as a
prediction made from padding. Requiring both handles either side with one expression.

**Dropping the rest.** Select with `where`. Multiplying by the mask is the obvious
alternative and it fails on real data: padding tokens are often banned from
generation by a `-inf` logit, the log-probability of "predicting" one is `-inf`, and
$0 \cdot (-\infty)$ is NaN.

**Perplexity** is the exponential of the average negative log-probability per scored
token, $\exp(-\text{logprob} / n)$: the geometric mean of $1 / p$. A model that spreads
its probability uniformly over $k$ tokens at every step has perplexity exactly $k$,
which is the easiest way to remember that the division by $n$ happens inside the
exponential.

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

JAX uses `jax.nn.log_softmax` and `jnp.take_along_axis`; MLX subtracts `mx.logsumexp`
and uses `mx.take_along_axis`. In PyTorch,
`-F.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none")`
is the same `picked` tensor in one call.

## Common mistakes

- **No shift.** Gathering `tokens[t]` from `logits[t]` scores each token under a
  distribution that was computed after reading it. The numbers look like
  log-probabilities and mean nothing.
- **Shifting the wrong way.** `logits[:, 1:]` with `tokens[:, :-1]`: now every
  prediction is two positions late.
- **`log(softmax(...))`.** Correct until a target is unlikely enough, then `-inf`, and
  the whole sequence scores `-inf`.
- **Ignoring the mask**, or masking only one side of the pair. Using `mask[:, :-1]`
  alone counts the prediction of the first padding token in a right-padded row;
  `mask[:, 1:]` alone counts the prediction of the first real token from padding in a
  left-padded row. Each variant passes every test on the other padding side.
- **Multiplying by the mask.** NaN as soon as an uncounted pair has log-probability
  `-inf`.
- **Averaging over the wrong count.** `mean` over all `T - 1` positions divides short
  rows by too much, so padding makes a row look better than it is. Dividing by the
  number of real tokens is off by one: `n` tokens give `n - 1` predictions.
- **`exp` of the sum.** That is $1 / p(\text{sequence})$, which grows exponentially
  with length and overflows for any real text. Perplexity is per token.

## Complexity

$O(B\,T\,V)$ time and memory for the `log_softmax`; the gather and the sums are
$O(B\,T)$.

## In the wild

Evaluation code spells the same shift as `shift_logits = logits[..., :-1, :]` and
`shift_labels = labels[..., 1:]`, followed by a cross-entropy that ignores a sentinel
label at padding positions. Reported perplexity is normally a **corpus-level** number:
the negative log-probabilities of all scored tokens are summed over the whole dataset
and divided by the total token count before the exponential. That is not the mean of
the per-row perplexities this function returns, because long rows carry more tokens
and because the exponential is not linear, so aggregate with `sequence_logprob` and
the pair counts, not with `perplexity`.

The same gather appears in serving. An API that returns `logprobs` reports the
log-probability of each sampled token from the logits one position earlier. Scoring a
fixed continuation ("which of these four answers is most likely?") is
`sequence_logprob` restricted to the continuation's positions. Speculative decoding
runs the large model once over the drafted tokens and reads, at every drafted
position, the probability it assigns to the token the draft model chose: the logits
at position `t` judge the draft's token at `t + 1`.
