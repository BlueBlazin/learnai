## The idea

Greedy speculative decoding combines two properties from earlier problems. First, a causal model's logits at any position depend only on tokens up to that position. One target call over the verified sequence `seq` plus $k'$ draft tokens `drafts` produces the target's greedy prediction after `seq`, after `seq + [d_1]`, and so on, which are the exact rows `verify_greedy` compares against `drafts`. Second, a KV cache lets a model extend a sequence while paying only for the newly added tokens.

The main challenge in the loop is tracking what each cache covers and keeping both caches consistent with the verified sequence `seq`. Store the number of tokens covered by each model's cache (`t_len` for the target, `d_len` for the draft), and apply two rules to both models:

- **Feed only unseen tokens.** Every call passes `(seq + drafts)[covered:]`, where `covered` is `t_len` for the target or `d_len` for the draft. For the target, this happens once per round and feeds any unseen verified tokens (after round 1, just the final token emitted by the previous round) together with all $k'$ drafts. For the draft model, this happens once per proposed token.
- **Roll back to all verified tokens except the newest one.** A round extends `seq` to `seq + d_1..d_n + g_n`, where $g_n$ is the target's correction or bonus token. After verification, the target's cache covers the old `seq` plus all $k'$ drafts, so any drafts after $d_n$ are invalid. Truncating the target cache to `len(seq) - 1` drops the rejected drafts and leaves out $g_n$. Leaving $g_n$ out is intentional, because feeding $g_n$ at the start of the next round produces the logits needed for the next position.

The draft cache follows the same bound with one twist. During drafting, the draft model feeds $d_1, \dots, d_{k'-1}$ to propose the next tokens, but it never feeds its final proposal $d_{k'}$ because nothing needed logits after $d_{k'}$. After a rejection, the draft cache is too long and gets truncated to `len(seq) - 1`, just like the target cache. After a fully accepted round, however, the draft cache is already one token shorter than `len(seq) - 1`, so the next round feeds two tokens ($d_{k'}$ and the bonus token). Using `min(d_len, len(seq) - 1)` handles both cases uniformly.

Correctness never depends on the draft model. Every emitted token is either a draft the target agreed with or the target's own argmax over a verified prefix, so the output always matches plain greedy decoding. The draft model affects only the number of rounds $R$.

The cost bounds follow directly from these two rules. In round 1, the target processes the prompt of length $P$ plus $k'$ drafts. In each later round, it processes at most $k' + 1$ tokens, for a total of at most $P - 1 + R(k + 1)$. The draft processes the prompt once and then at most $k + 1$ tokens per round. Without truncation-based rollback, removing rejected tokens would require rebuilding the cache over the full sequence each round.

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

Setting `n_draft = min(k, remaining - 1)` ensures each round emits at most `remaining` tokens, so the loop never overshoots `max_new_tokens` and generates the last token with the target alone when only one token remains. The stopping check runs before rollback so neither cache is truncated after the final round. The JAX and MLX solutions differ only in the argmax calls.

## Common mistakes

- **Not rolling back the target's cache.** Leaving rejected drafts in `t_cache` causes the target to condition on tokens that were never emitted. The generated text stays fluent but no longer matches greedy decoding.
- **Not rolling back the draft's cache.** Because the target still verifies every token, the output remains exact. However, the draft model now predicts from a context containing rejected tokens, acceptance collapses, and target calls approach one per token.
- **Ignoring an EOS inside the accepted run.** Checking only `new[-1] == eos_id` misses an `eos_id` that appeared among the accepted drafts, causing generation to continue past the stop token.
- **Overshooting `max_new_tokens`.** A round can produce up to `k + 1` tokens, so appending them all near the end of the budget returns too many tokens. Clamp `n_draft` as shown above, or trim `new` to the remaining budget.
- **Rebuilding a cache instead of truncating it.** Calling a model on `seq + drafts` with `cache=None` produces exact tokens, but costs $O(L)$ tokens per round for a sequence of length $L$ instead of $O(k)$. On real models, prefill work then dominates and speculation becomes slower than standard decoding.
- **Dropping the bonus token.** When all drafts are accepted, the target's final logit row is valid. Discarding it costs an extra round for every fully accepted block, losing a fifth of the speedup at $k = 4$ with a strong draft model.
- **Verifying only one draft per round.** Stopping the comparison after the first draft still produces exact tokens, but defeats the purpose of drafting a block and exceeds the target call limit.

## Complexity

Generating $g$ tokens in $R$ rounds takes $R$ target forward calls over at most $P + R(k + 1)$ tokens, and at most $R\,k$ draft forward calls over at most $P + R(k + 1)$ tokens. The round count $R$ ranges from $\lceil g / (k + 1) \rceil$ for a perfect draft to $g$ for a draft that never agrees. With a per-token agreement rate $\alpha$, each round produces $(1 - \alpha^{k+1}) / (1 - \alpha)$ tokens on average. Cache truncation is $O(1)$ here because the kit slices views, and is similarly a pointer update or slice in production engines.

## In the wild

This loop is the greedy special case of Algorithm 1 in Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023, arXiv:2211.17192) and Algorithm 2 in Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023, arXiv:2302.01318). Both algorithms draft $K$ tokens, score $K + 1$ positions in parallel, keep the accepted prefix, and append one token from the target. Under one-hot (greedy) distributions, the acceptance test $\min(1, p/q)$ reduces to checking argmax equality, and the residual distribution reduces to the target's argmax.

Production engines handle cache rollback the same way. Hugging Face `transformers` crops both models' `past_key_values` to the accepted length after each round of assisted generation, and llama.cpp removes rejected positions from both KV caches by sequence index. Full serving systems extend this loop with exact stochastic verification (`speculative-sampling-block`), adaptive draft lengths based on recent acceptance rates, ragged batching where each sequence accepts a different count, and alternative drafters such as n-gram lookup or the target's own early layers.
