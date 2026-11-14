## The idea

A step's duration grows with the number of tokens in it. Fix a budget $B$ of tokens
per step and the duration of every step is bounded, here by
$8 + 0.25\,P + D$ ms with $P + D \le B$. If every running sequence takes part in
every step, the time between two of its tokens is at most one step. That is the whole
latency argument, and it dictates the order in which the budget is spent:

1. **Decodes are paid first.** They are cheap (one token each) and they are the ones
   with a user watching. Paying prompts first lets a queue of prompts push decodes out
   of the step, which is the stall the budget was supposed to prevent.
2. **Prompts take the rest, in chunks.** A prompt has no natural size limit, so it
   must be divisible: it takes what is left now and comes back for more. Its first
   token appears when the last chunk is done; the chunks before it produce nothing
   visible, they only fill the KV cache.

The threshold adds fairness between prompts. Without it, first come, first served
means the prompt at the head takes everything the decodes leave, and a 20-token
question that arrives behind a 4,000-token document waits for all of its chunks. With
a cap of $T$ tokens per sequence per step, the long prompt still advances every step,
and the remainder of the budget flows to the requests behind it.

Because each candidate receives $\min(\text{remaining}, T, \text{budget left}) \ge 1$
tokens while budget remains, nobody is skipped, the order is strictly first come,
first served, and the budget is exhausted exactly when there is enough demand: the
planner is work-conserving.

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

In the stall scenario of the statement (six running chats, a 4,000-token prompt), the
longest gap between tokens is 140.5 ms with the threshold at the budget and 64 ms with
a threshold of 128, where the short prompts behind the long one get their first token
after 70 ms instead of 1,151 ms.

## Common mistakes

- **Prefill first.** Filling the budget with prompt chunks and decoding with what is
  left starves the running sequences whenever prompts can absorb the budget: 1,088 ms
  between tokens in the stall scenario, the very freeze chunking is meant to remove.
- **The whole prompt in one step.** Scheduling a prompt unsplit either exceeds the
  budget (the engine refuses the step) or, in the variant that waits until the prompt
  fits, never schedules a prompt longer than the budget and blocks the queue behind
  it forever.
- **Decoding a sequence with its last chunk.** The chunk that completes the prompt is
  the forward pass that produces the first token. Listing the sequence in `decode` as
  well asks for two tokens from one pass; the engine rejects it as a duplicate.
- **Not charging the decodes.** Giving prompts the full `max_batch_tokens` on top of
  the decodes exceeds the budget by the number of running sequences. It goes unnoticed
  until the server is busy, which is when the budget matters.
- **Ignoring the threshold.** Capping a chunk only by the budget is correct chunking
  and bad head-of-line behavior: short prompts queue behind every chunk of a long one.
- **Shortest prompt first.** Sorting the candidates looks like an optimization and
  starves long prompts under a steady stream of short ones. If long prompts should
  yield, the threshold does that without ever stopping them.

## Complexity

$O(|\text{running}| + k)$ where $k$ is the number of candidates that receive a chunk,
at most the budget. The planner runs once per step, so it must stay far below the
step's own duration; this one is a few hundred nanoseconds per entry.

## In the wild

- **Sarathi-Serve** (Agrawal et al., OSDI 2024) names the two ideas: *chunked
  prefills* split a prompt over iterations, and *stall-free batching* builds each
  batch from all running decodes first, then partially prefilled requests, then new
  ones, under a per-iteration token budget. The budget is derived from the
  time-between-tokens target: the largest batch whose iteration time still meets it.
- **vLLM** schedules the same way with chunked prefill enabled (the default in its
  current engine): running requests are scheduled first, then waiting ones, each
  taking `min(tokens needed, token budget)` of `max_num_batched_tokens`.
  `long_prefill_token_threshold` caps the tokens a single request may take per step
  exactly as here, and `max_num_partial_prefills` and `max_long_partial_prefills`
  additionally limit how many prompts may be in progress at once, because every
  partially prefilled prompt holds KV memory without producing tokens.
- **TGI** bounds the batch with `max_batch_prefill_tokens` and
  `max_batch_total_tokens`, and **SGLang** exposes the chunk size directly
  (`--chunked-prefill-size`).
- The budget is a throughput knob as much as a latency one: a small budget makes steps
  short and prompts slow (more steps, each paying the fixed cost), a large one the
  opposite. Separating prefill and decode onto different machines removes the trade
  altogether (`prefill-decode-disaggregation`).
