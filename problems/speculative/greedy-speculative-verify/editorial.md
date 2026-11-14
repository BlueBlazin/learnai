## The idea

With greedy decoding the target model's output is a deterministic function of the
prefix: the next token is the argmax of its logits. A draft token is therefore
"correct" exactly when it equals that argmax, and speculative decoding reduces to
checking guesses.

The target is run once on the verified sequence plus the $K$ draft tokens. A causal
model's output at a position depends only on the tokens up to it, so that single pass
yields, for every $i$, the logits the target *would have produced* had it been fed
the sequence up to `draft_tokens[i-1]` alone. Row $i$ is trustworthy if and only if
the context it was computed from is the true greedy sequence, that is, if draft
tokens $0..i-1$ were all correct:

- Row 0 depends on no draft token. It is always valid, so the round always yields at
  least one token.
- If rows $0..i-1$ agree with the draft, row $i$ is valid too (induction).
- At the first mismatch $n$, row $n$ is still valid, and it tells us the right token
  for that position. Rows after $n$ were computed from a context containing the wrong
  token and are worthless, whatever they say.
- If nothing mismatches, row $K$ is valid, and it is one more token for free.

So the answer is the length $n$ of the matching prefix and the target's token in row
$n$. One target pass produces between 1 and $K + 1$ tokens, and the output sequence
is identical to what the target alone would have generated.

## Reference solution (PyTorch)

```python
def verify_greedy(draft_tokens, target_logits):
    greedy = torch.argmax(target_logits, dim=-1)
    match = (greedy[:-1] == draft_tokens).to(torch.int64)
    n_accepted = int(torch.cumprod(match, dim=0).sum())
    return n_accepted, int(greedy[n_accepted])
```

The cumulative product of the 0/1 match vector is 1 up to the first mismatch and 0
from there on, so its sum is the prefix length. `greedy[n_accepted]` covers both
outcomes without a branch: the correction when `n_accepted < K`, the bonus token
when `n_accepted == K`. The JAX and MLX versions are the same four lines. A Python
loop that breaks at the first mismatch is equally good for one sequence; the
vectorized form is what generalizes to a batch.

## Common mistakes

- **Counting all matches.** `match.sum()` accepts a position that matches after an
  earlier mismatch. That token was predicted from a context containing a rejected
  token; accepting it makes the output differ from the target's greedy decode.
- **Shifted alignment.** In an ordinary decoding loop "the logits for the next token"
  are the last row, and it is tempting to pair `draft_tokens[i]` with row `i + 1`.
  Row `i` is the one that scores `draft_tokens[i]`. With the shift almost nothing is
  accepted, and what is accepted is wrong.
- **No bonus token.** Slicing the logits to `K` rows "to match the draft" throws away
  the one row that is valid whenever everything was accepted. The loop stays correct
  only if it then spends another target pass on a token it already had.
- **Taking the next token from the last row.** `logits[-1]` is right only after a
  full match. After a rejection at $n < K$ the correction is in row $n$.
- **All or nothing.** Rejecting the whole block on any mismatch is still lossless but
  discards the valid prefix, and with it most of the speedup: with a per-token
  agreement rate of 0.8 and $K = 4$, a block survives intact only 41% of the time.

## Complexity

$O(K V)$ time for the argmax, $O(K)$ extra memory.

## In the wild

Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative
Decoding" (ICML 2023, arXiv:2211.17192) and Chen et al., "Accelerating Large Language
Model Decoding with Speculative Sampling" (2023, arXiv:2302.01318) define
verification for sampling: accept draft token $x$ with probability
$\min(1, p(x)/q(x))$ and resample from the residual on rejection. Greedy decoding is
the limit in which both distributions are one-hot: the ratio is 1 when the tokens
agree and 0 otherwise, and the residual is the target's one-hot distribution, which
gives exactly the rule above. Hugging Face `transformers` (assisted generation),
vLLM, and llama.cpp all implement this comparison batched over sequences, with the
cumulative-product (or "first mismatch") trick so that rows can accept different
numbers of tokens. The drafts need not come from a model: prompt lookup, an n-gram
cache, or the multi-token-prediction heads of the target itself feed the same
verifier.
