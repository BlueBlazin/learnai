## The idea

Greedy speculative decoding rests on two facts from earlier problems. A causal
model's logits at a position depend only on the tokens up to it, so one target call
over `S + drafts` yields the target's greedy token after `S`, after `S + d_1`, and
so on: the rows that `verify_greedy` compares with the drafts. And a KV cache makes
continuing a sequence cost only the new tokens. The loop combines them, and its
whole difficulty is one question: **what does each cache cover, and is all of it
still true?**

Keep the verified sequence `seq` and, per model, the number of its tokens the cache
covers. Then both models follow the same two rules.

- **Feed what is missing.** A call passes `(seq + drafts)[covered:]`. For the
  target that is one call per round: the verified tokens it has not seen (after
  round 1, exactly one: the token the previous round ended with) plus all drafts.
  For the draft it is one call per proposal.
- **After the round, keep only what is verified, minus the newest token.** The
  round turned `seq` into `seq + d_1..d_n + g_n`. The target's cache covers the old
  `seq` and all $k'$ drafts; drafts after $d_n$ are wrong. Truncating to
  `len(seq) - 1` removes them. It also leaves out $g_n$, which no model has been fed
  yet, and that is exactly right: feeding it is what produces the logits the next
  round starts from.

The draft's cache obeys the same bound with one twist. Within a round the draft
feeds $d_1, \dots, d_{k'-1}$ but never $d_{k'}$ (nothing needed its logits). After a
rejection its cache is too long and is truncated like the target's. After a fully
accepted round it is one token *short* of `len(seq) - 1`, and the next call feeds
two tokens, $d_{k'}$ and the bonus token. `min(d_len, len(seq) - 1)` covers both
cases.

Correctness does not depend on the draft at all: every generated token is either a
draft the target agreed with or the target's own argmax, computed from a context of
verified tokens. That is why the tests can compare with plain greedy decoding. The
draft only determines $R$, the number of rounds.

The cost follows from the rules. The target sees the prompt and $k'$ drafts in
round 1 and $k' + 1$ tokens per later round: at most $P - 1 + R(k + 1)$. The draft
sees the prompt, then at most $k + 1$ tokens per round. Without rollback by
truncation, the only way to remove a rejected token from a cache is to rebuild it,
which costs the whole sequence every round.

## Reference solution (PyTorch)

```python
def speculative_greedy(target, draft, prompt, k, max_new_tokens, eos_id):
    seq = list(prompt)
    t_cache = d_cache = None
    t_len = d_len = 0
    while True:
        n_draft = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)

        drafts = []
        for _ in range(n_draft):
            logits, d_cache = draft((seq + drafts)[d_len:], d_cache)
            d_len = len(seq) + len(drafts)
            drafts.append(int(torch.argmax(logits[-1])))

        logits, t_cache = target((seq + drafts)[t_len:], t_cache)
        greedy = torch.argmax(logits[-(n_draft + 1):], dim=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        new = [*drafts[:n], greedy[n]]
        if eos_id in new:
            new = new[: new.index(eos_id) + 1]
        seq += new
        if new[-1] == eos_id or len(seq) - len(prompt) == max_new_tokens:
            return seq[len(prompt):]

        t_len = len(seq) - 1
        t_cache = t_cache.truncate(t_len)
        d_len = min(d_len, len(seq) - 1)
        d_cache = d_cache.truncate(d_len)
```

`n_draft = min(k, remaining - 1)` makes a round yield at most `remaining` tokens, so
the budget can never be overshot, and the final token of a generation is produced by
the target alone. The stop test comes before the rollback, so no cache is touched
after the last round. The JAX and MLX versions differ only in the argmax calls.

## Common mistakes

- **Not rolling back the target's cache.** After a rejection the cache still holds
  the rejected drafts. Feeding the correction on top of them makes the target
  predict from a sequence that was never generated. The output is fluent and wrong.
- **Not rolling back the draft's cache.** The mirror image, and harder to notice:
  the target still verifies everything, so the output is right. But the draft now
  proposes from a context containing its rejected tokens, almost nothing is
  accepted, and the loop needs close to one target call per token.
- **Ignoring an EOS inside the accepted run.** Checking only the last token of a
  round misses an EOS among the accepted drafts, and generation continues past the
  end of the answer.
- **Overshooting `max_new_tokens`.** A round yields up to `k + 1` tokens; appending
  them all near the end of the budget returns too many. Either clamp the draft
  length, as here, or cut the round's tokens to the budget.
- **Rebuilding a cache instead of truncating it.** Calling a model on
  `seq + drafts` without a cache is always correct and costs $O(L)$ tokens per call
  instead of $O(k)$. On a real model the target's prefill then dominates and
  speculation is slower than plain decoding.
- **Dropping the bonus token.** The target's last row is valid whenever every draft
  was accepted. Emitting only the drafts wastes it and costs an extra round for
  every fully accepted block: with a good draft, a fifth of the speedup at $k = 4$.
- **Verifying one draft per round.** Stopping the comparison after the first draft
  is still exact, and throws away the reason for drafting a block.

## Complexity

With $g$ generated tokens and $R$ rounds: $R$ target calls over at most
$P + R(k + 1)$ tokens, and at most $R\,k$ draft calls over at most $P + R(k + 1)$
tokens. $R$ is between $\lceil g / (k + 1) \rceil$ (perfect draft) and $g$ (useless
draft); with a per-token agreement rate $\alpha$ a round yields
$(1 - \alpha^{k+1}) / (1 - \alpha)$ tokens on average. Truncation is $O(1)$ here
(the kit slices views) and a pointer move or a slice in a real engine.

## In the wild

The scheme is the greedy special case of Leviathan, Kalman, and Matias, "Fast
Inference from Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192,
Algorithm 1) and of Chen et al., "Accelerating Large Language Model Decoding with
Speculative Sampling" (2023, arXiv:2302.01318, Algorithm 2), both of which draft
$K$ tokens, score $K + 1$ positions in parallel, keep the accepted prefix, and add
one token from the target. With one-hot distributions their acceptance test
$\min(1, p/q)$ becomes "the argmaxes agree" and the residual becomes the target's
argmax, which is the rule used here.

Engines implement the rollback the same way. `transformers` crops both models'
`past_key_values` to the accepted length after each round of assisted generation;
llama.cpp removes the rejected positions from the KV cache of the draft and target
contexts by sequence position. Production loops add what this problem leaves
out: sampling with the exact accept-or-resample rule (`speculative-sampling-block`),
a draft length adapted to the recent acceptance rate, batches in which every row
accepts a different number of tokens, and drafts that come from an n-gram lookup or
from the target's own early layers instead of a second model.
