## The idea

A step's duration grows with the number of tokens it processes. Let $B$ be the per-step token budget (`max_batch_tokens`), $P$ be the number of prompt tokens in a step, and $D$ be the number of decoded sequences. A step takes $8 + 0.25\,P + D$ ms, so enforcing $P + D \le B$ bounds every step's duration to at most $8 + B$ ms. If every running sequence participates in every step, the time between two consecutive tokens of any running sequence never exceeds one step.

That latency bound determines the order in which the budget is spent:

1. **Pay decodes first.** Decodes are cheap (1 token each) and directly serve users watching streaming output. Spending budget on prompts first lets a queue of prompts push decodes out of the step, reintroducing the stalls that the budget is meant to prevent.
2. **Fill the remaining budget with prompt chunks.** Prompts have no small upper bound on length, so they must be divisible. A prompt takes whatever budget is available now and resumes in later steps. Intermediate chunks only populate the KV cache, and the first output token appears when the final chunk completes.

Let $T$ be `long_prefill_threshold`. Without $T$, strict first-come, first-served scheduling allows the head prompt to consume the entire remaining budget, forcing a 20-token question behind a 4,000-token document to wait for every chunk of that document. Capping each prompt at $T$ tokens per step lets the long prompt advance every step while passing leftover budget to requests behind it.

While budget remains, each candidate receives $\min(\text{remaining}, T, \text{budget left}) \ge 1$ tokens. Nobody is skipped, admission order stays strictly first-come, first-served, and the budget is fully used whenever demand exists, so the planner is work-conserving.

## Reference solution (Python)

```python
def plan(running, prefilling, waiting, max_batch_tokens, long_prefill_threshold):
    decode = list(running[:max_batch_tokens])
    budget = max_batch_tokens - len(decode)

    prefill = []
    for seq_id, remaining in [*prefilling, *waiting]:
        if budget == 0:
            break
        n = min(remaining, long_prefill_threshold, budget)
        prefill.append((seq_id, n))
        budget -= n
    return prefill, decode
```

In the stall scenario from the statement (six running sequences and a 4,000-token prompt), the longest inter-token gap is 140.5 ms when `long_prefill_threshold` equals the budget, and 64 ms when `long_prefill_threshold = 128`. With the 128-token cap, the short prompts behind the long one get their first token in 70 ms instead of 1,151 ms.

## Common mistakes

- **Prefill first.** Spending the budget on prompt chunks before decodes starves running sequences whenever prompts can fill the budget. In the stall scenario, this stretches the inter-token gap to 1,088 ms.
- **Scheduling whole prompts in one step.** Putting an unsplit prompt into the batch exceeds the budget and causes the engine to reject the step. Waiting until the prompt fits never schedules any prompt longer than `max_batch_tokens` and blocks the queue behind it forever.
- **Decoding a sequence alongside its last prompt chunk.** The chunk that finishes a prompt already produces its first output token. Listing the same sequence in `decode` asks for two tokens in one forward pass, which the engine rejects as a duplicate.
- **Not charging decodes against the budget.** Giving prompts the full `max_batch_tokens` on top of decodes exceeds the budget by `len(decode)`. This bug stays hidden when the server is idle and breaks under load.
- **Ignoring `long_prefill_threshold`.** Capping chunks only by the remaining budget prevents decode stalls but leaves head-of-line blocking intact, because short prompts wait behind every chunk of a long prompt.
- **Shortest prompt first.** Sorting candidates by remaining length starves long prompts under a steady stream of short requests. Using `long_prefill_threshold` lets short prompts share the step without ever halting long prompts.

## Complexity

Time complexity is $O(|\text{running}| + k)$, where $k$ is the number of candidates that receive a chunk (at most `max_batch_tokens`). Because the planner runs before every step, it must stay far cheaper than the step itself. This implementation takes only a few hundred nanoseconds per entry.

## In the wild

- **Sarathi-Serve** (Agrawal et al., OSDI 2024) introduced *chunked prefills* to split prompts across iterations and *stall-free batching* to pack all running decodes first, then partially prefilled requests, then waiting requests under a per-iteration token budget. Operators choose the budget as the largest batch whose step time still meets the inter-token latency target.
- **vLLM** uses the same scheduling order by default when chunked prefill is enabled. Running requests go first, followed by waiting requests taking `min(tokens needed, token budget)` out of `max_num_batched_tokens`. Its `long_prefill_token_threshold` parameter caps per-step prompt tokens just like `long_prefill_threshold` here, while `max_num_partial_prefills` and `max_long_partial_prefills` also limit how many prompts can be partially prefilled at once so KV cache memory is not tied up without producing tokens.
- **TGI** bounds batch size with `max_batch_prefill_tokens` and `max_batch_total_tokens`, and **SGLang** exposes the chunk limit directly through `--chunked-prefill-size`.
- The token budget trades throughput against latency. A small budget keeps steps short but slows down prefills by paying fixed per-step overhead more often, while a large budget does the reverse. Separating prefill and decode onto different machines removes this trade-off (`prefill-decode-disaggregation`).
