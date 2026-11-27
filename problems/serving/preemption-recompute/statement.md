Extend the continuous-batching scheduler loop to an engine whose KV cache is a finite pool of blocks. Because each running sequence grows by one token per step, a batch that fits at admission can run out of memory later. Your loop must reclaim memory by preempting a running sequence and later resuming it by recomputing its keys and values. The starter code is the loop from `continuous-batching-scheduler`, which does not track memory.

![This problem is the Engine loop, here with a KV cache that can run out of blocks.](assets/map.svg)

## Specification

```python
async def run(engine, queue: asyncio.Queue, max_running: int, watermark_blocks: int) -> None: ...
```

`run` never returns. The `queue` delivers [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects at their arrival times. Read `request.id` and `request.prompt_tokens`, and do not read `request.output_tokens`. Time is virtual and measured in **milliseconds**. A step that prefills $P$ tokens and decodes $D$ sequences takes $8 + 0.25\,P + D$ ms.

**Memory.** The [`SimEngine`](../../../python/src/learnai/kit/simengine.py) manages `engine.kv_blocks` blocks, each holding `engine.block_tokens` tokens. A sequence with $k$ cached tokens (queried via `engine.kv_tokens(id)`) occupies $\lceil k / \text{block\_tokens} \rceil$ blocks. When a step starts, prefilling $n$ tokens adds $n$ to $k$, and decoding adds $1$. When a step ends, any sequence that finishes frees all of its blocks.

After its prefill, a sequence holds its prompt tokens. Each subsequent decode step adds the token produced by the previous step. Therefore, a decoding sequence needs **one new block in a step if and only if `engine.kv_tokens(id) % engine.block_tokens == 0`**. If a step requires more blocks than are free, the engine raises `OutOfBlocks`. Like any other `ProtocolViolation`, raising `OutOfBlocks` fails the test case even if you catch it.

In addition to `admit` and `step`, you will use the following engine properties and methods:

- `engine.free_blocks`, `engine.kv_tokens(id)`, and `engine.block_tokens` report current block availability, cached token count, and block size.
- `engine.preempt(id)` immediately frees all blocks held by that sequence. The sequence remains registered in the engine and **keeps the output tokens it has already produced**. To resume it later, you must prefill its entire context in a single step using `engine.remaining_prefill(id)` tokens (its original prompt plus all output tokens generated so far). That recompute step produces the sequence's *next* output token, and the sequence decodes normally on subsequent steps. Do not call `engine.admit` on a preempted sequence again.

**The policy.** *Waiting* requests hold no KV blocks. *Running* sequences hold KV blocks and are ordered by when they were last admitted, oldest first. Before every step, follow these four stages in order:

1. **Collect arrivals.** Drain all currently available requests from `queue` and append them to the back of the waiting queue in arrival order.
2. **Allocate memory to running sequences, oldest first.** Track how many blocks are promised in the current step, starting at 0. A block is *available* if it is free in the engine and not yet promised. Iterate through the running sequences from oldest to newest. If a sequence needs a new block and no block is available, **preempt the most recently admitted running sequence** by calling `engine.preempt`, removing it from the running list, and pushing it to the **front** of the waiting queue. Repeat until a block is available or the sequence being served has itself been preempted (because it was the most recent). If the sequence is still running, promise it a block (if it needs one) and include it in this step's decode list.
3. **Admit waiting requests, first come, first served, only if stage 2 preempted nothing.** While fewer than `max_running` sequences are running and the waiting queue is non-empty, inspect the front request. Its prefill length is `request.prompt_tokens` for a new request, or `engine.remaining_prefill(id)` for a preempted one, requiring `needed` $= \lceil \text{prefill} / \text{block\_tokens} \rceil$ blocks. Admit it if `available - needed >= watermark_blocks`. As an exception, if nothing is decoding in this step and nothing has been admitted yet, admit it as long as `available - needed >= 0`. When admitting a request, promise its blocks, append it to the running list, and call `engine.admit` if it has never been admitted before. If the front request cannot be admitted, stop admission immediately so no later request overtakes it.
4. **Execute one step.** Call `engine.step` with `prefill` containing the sequences admitted in stage 3 (in admission order, each with its full prefill length) and `decode` containing the sequences from stage 2 (oldest first).

When the step finishes, remove completed sequences from the running list and plan the next step at the same instant. If nothing is waiting and nothing is running, wait for the next arrival with `await queue.get()`. Arrivals occurring at the exact instant a step finishes are already in `queue` before the next step is planned, and simultaneous arrivals are queued in order, matching `continuous-batching-scheduler`.

Notice two consequences of this policy. First, multiple sequences may be preempted in a single step, and pushing each victim to the front of the waiting queue preserves their original running order. Second, a preempted sequence never fits back into the same step that preempted it, so stage 3's no-preemption guard simply makes explicit what the block count already guarantees.

## Example

Suppose `max_running = 3`, `watermark_blocks = 1`, and the KV pool has 7 blocks of 4 tokens each. Four requests arrive, written as `(arrival, prompt_tokens, output_tokens)`: request 0 `(0, 8, 6)`, request 1 `(0, 8, 4)`, request 2 `(4, 6, 5)`, and request 3 `(30, 4, 2)`.

| starts | prefill | decode | blocks held during the step (0, 1, 2, 3) | notes |
|---|---|---|---|---|
| 0 | `(0, 8), (1, 8)` | | 2, 2, -, - | 3 blocks remain free |
| 12 | | `0, 1` | 3, 3, -, - | Both cross a block boundary, leaving 1 block available. Request 2 needs 2 blocks ($1 - 2 < 1$) |
| 22 | | `0, 1` | 3, 3, -, - | Request 2 still cannot be admitted ($1 - 2 < 1$) |
| 32 | | `0, 1` | 3, 3, -, - | Request 1 finishes at 42 ms |
| 42 | `(2, 6), (3, 4)` | `0` | 3, -, 2, 1 | 4 blocks available. Request 2 leaves 2 blocks and request 3 leaves 1, both meeting the watermark |
| 53.5 | | `0, 2` | 4, -, 2, - | Request 0 (12 tokens) takes the last block. Request 3 (4 tokens) needs a block, none is available, and request 3 is the most recently admitted sequence, so it is preempted after 1 output token |
| 63.5 | `(3, 5)` | `2` | -, -, 2, 2 | Request 0 finished. Request 3 resumes by prefilling its 4 prompt tokens plus its 1 output token (5 tokens total) |
| 73.75 | | `2` | -, -, 3, - | Request 3 finished with its second output token |
| 82.75 | | `2` | -, -, 3, - | Request 2 finishes at 91.75 ms |

![The worked example, one row per request. Request 3 is preempted at 53.5 ms, waits first in line, and returns at 63.5 ms with a 5-token prefill.](assets/preempt-timeline.svg)

Request 3 produced token 0 at 53.5 ms and token 1 at 73.75 ms without restarting from scratch. The 5 tokens prefilled at 63.5 ms are the cost of the preemption (*recomputed tokens*). The watermark of 1 kept request 2 out at 12 ms and 22 ms, when admitting it would have exhausted the pool, though it could not prevent the preemption at 53.5 ms when two sequences crossed a block boundary in the same step.

## Constraints

- `1 <= max_running <= 16`, `0 <= watermark_blocks < kv_blocks`, and block sizes range from 1 to 16 tokens. For every request, `prompt_tokens + output_tokens - 1` tokens fit in the pool, so a sequence running alone always finishes. Arrival times are multiples of 0.25 ms.
- The policy uniquely determines every step and every preemption. Tests compare both exactly against literal schedules on small scenarios and against the reference implementation on seeded bursty and Poisson workloads of up to 120 requests. Tests also verify from the trace that every request finishes, each output token is produced exactly once in order, no sequence is released or admitted twice, every preempted sequence is the most recently admitted running sequence, and at most `max_running` sequences hold memory at once.
- **Cost.** On a Poisson workload of 120 requests (prompts of 16 to 160 tokens, outputs of 8 to 96 tokens) with `max_running = 8`, 40 blocks of 16 tokens, and `watermark_blocks = 4`, at most **3000** tokens may be recomputed (`engine.counters.recomputed_tokens`). The reference recomputes 1759 tokens across 16 preemptions, compared to 5527 tokens across 63 preemptions without the watermark.
- Use `asyncio` and the engine only.

## Why this matters

A serving engine cannot know in advance how long a generation will be, so it cannot reserve memory for the final length up front. Reserving the maximum possible context for every sequence wastes most of the KV cache, while optimistic admission fills the GPU and eventually runs out of blocks mid-batch. When that happens, preempting the newest sequence wastes the least work and preserves first-come, first-served ordering. Keeping its generated tokens and recomputing only its KV cache makes the preemption invisible to the client aside from a brief pause.

Small implementation mistakes cause quiet failures in production. Restarting a victim streams duplicate tokens to the client, picking the oldest sequence causes starvation, and omitting a watermark leads to thrashing where the GPU spends most of its time recomputing the same prompts while throughput collapses. Preemption count is one of the first metrics engineers check when a serving deployment slows down under load.
