## The idea

Under greedy decoding, the target model's next token is a deterministic function of
the prefix, namely the argmax of its logits. A draft token is therefore correct if
and only if it equals that argmax, so speculative verification reduces to checking
guesses.

The target model runs once on the verified prefix plus the $K$ draft tokens. Because
a causal model's output at any position depends only on earlier tokens, that single
pass produces, for every $i$, the exact logits the target would have output if fed
the prefix up to `draft_tokens[i-1]`. Row $i$ is valid if and only if its input
context matches the true greedy sequence, meaning draft tokens `0..i-1` were all
correct:

- Row 0 depends on no draft tokens, so it is always valid. Every round therefore
  produces at least one token.
- By induction, if rows `0..i-1` all match the draft, then row $i$ is valid too.
- At the first mismatch $n$, row $n$ is still valid and supplies the correct token
  for position $n$. Every row after $n$ was conditioned on the wrong token and must
  be discarded, even if its argmax happens to agree with the draft.
- If all $K$ draft tokens match, row $K$ is valid and provides one bonus token for
  free.

The result is therefore the length $n$ of the matching prefix together with the
target's greedy token from row $n$. Each target pass yields between 1 and $K + 1$
tokens, and the final sequence is identical to greedy decoding with the target alone.

## Reference solution (PyTorch)

```python
def verify_greedy(draft_tokens, target_logits):
    greedy = torch.argmax(target_logits, dim=-1)
    match = (greedy[:-1] == draft_tokens).to(torch.int64)
    n_accepted = int(torch.cumprod(match, dim=0).sum())
    return n_accepted, int(greedy[n_accepted])
```

The cumulative product of the 0/1 match vector stays 1 up to the first mismatch and
becomes 0 from that point onward, so summing it gives the accepted prefix length.
Indexing `greedy[n_accepted]` handles both outcomes without a branch. It selects the
correction token when `n_accepted < K`, and it selects the bonus token when
`n_accepted == K`. The JAX and MLX implementations use the same four lines. A plain
Python loop that breaks on the first mismatch works just as well for a single
sequence, while the vectorized form extends directly to batches.

## Common mistakes

- **Counting all matches.** Using `match.sum()` accepts positions that match after an
  earlier mismatch. Those logits were computed from a context containing a rejected
  token, so accepting them causes the output to diverge from the target model.
- **Shifted alignment.** Because standard decoding reads next-token logits from the
  last row, it is easy to pair `draft_tokens[i]` with row `i + 1`. Row `i` is the row
  that scores `draft_tokens[i]`. Shifting by one rejects almost everything and
  accepts wrong tokens when it does match.
- **Dropping the bonus token.** Slicing `target_logits` to `K` rows to match the
  length of `draft_tokens` discards row `K`, which is valid whenever all drafts are
  accepted. The decoder then wastes an extra target pass to recompute a token it
  already had.
- **Taking `next_token` from the last row.** `target_logits[-1]` is only valid when
  all `K` draft tokens match. After a mismatch at $n < K$, the correction comes from
  row $n$.
- **All-or-nothing rejection.** Discarding the whole block on any mismatch remains
  exact, but it throws away the valid prefix and most of the speedup. With a
  per-token agreement rate of 0.8 and $K = 4$, an entire block survives intact only
  41% of the time.

## Complexity

$O(K V)$ time for the argmax over `target_logits`, and $O(K)$ extra memory, where $V$
is the vocabulary size.

## In the wild

Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative
Decoding" (ICML 2023, arXiv:2211.17192), and Chen et al., "Accelerating Large
Language Model Decoding with Speculative Sampling" (2023, arXiv:2302.01318), define
verification for sampling with target distribution $p$ and draft distribution $q$.
They accept draft token $x$ with probability $\min(1, p(x) / q(x))$ and resample from
the residual on rejection. Greedy decoding is the special case where both
distributions are one-hot. The acceptance ratio is 1 when the tokens agree and 0
otherwise, and the residual is the target's one-hot distribution, which yields the
exact rule above.

Hugging Face `transformers` (assisted generation), vLLM, and llama.cpp all run this
verification batched over sequences using the cumulative-product (or first-mismatch)
trick so each sequence in a batch can accept a different number of tokens. Drafts do
not even require a separate model. Prompt lookup, n-gram caches, and the target
model's own multi-token-prediction heads all feed the same verifier.
