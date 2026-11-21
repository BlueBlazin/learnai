Extend the continuous-batching loop to an engine whose KV cache is a finite pool
of blocks. Sequences grow by one token per step, so a batch that fit when it was
admitted can run out of memory later; the loop must then take memory away from a
running sequence, and give it back later by computing that sequence's keys and
values again. The starter is the loop of `continuous-batching-scheduler`, which
knows nothing about memory.

![This problem is the Engine loop, here with a KV cache that can run out of blocks.](assets/map.svg)

## Specification

```python
async def run(engine, queue: asyncio.Queue, max_running: int, watermark_blocks: int) -> None: ...
```

`run` never returns. `queue` receives
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects at their
arrival times; read `request.id` and `request.prompt_tokens`, not
`request.output_tokens`. Time is virtual, in **milliseconds**: a step over $P$
prefilled tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms.

**Memory.** The [`SimEngine`](../../../python/src/learnai/kit/simengine.py) has
`engine.kv_blocks` blocks of `engine.block_tokens` tokens. A sequence that holds
$k$ tokens of keys and values occupies $\lceil k / \text{block\_tokens} \rceil$
blocks. Prefilling $n$ tokens adds $n$ to $k$ and decoding adds 1, **when the step
starts**; a sequence that finishes frees its blocks when the step ends. After its
prefill a sequence holds its prompt; each decode step adds the token produced by
the step before. So a decoding sequence needs **one new block in a step exactly
when `engine.kv_tokens(id) % engine.block_tokens == 0`**. A step that needs more
blocks than are free is refused with `OutOfBlocks`, and that, like every other
`ProtocolViolation`, fails the case even if it is caught.

What you need from the engine, besides `admit` and `step`:

- `engine.free_blocks`, `engine.kv_tokens(id)`, `engine.block_tokens`.
- `engine.preempt(id)`: frees the sequence's blocks at once. The sequence stays
  known to the engine and **keeps the output tokens it has produced**. To continue
  it, its whole context must be prefilled again in one step:
  `engine.remaining_prefill(id)` tokens, which is its prompt plus the tokens it has
  produced. The step that does so produces its *next* token, and from the step
  after that it decodes as before. Do not `admit` it again.

**The policy.** *Waiting* requests hold no memory; *running* sequences do, and are
kept in the order in which they were last admitted. Before every step:

1. **Collect** arrivals from `queue` to the back of the waiting queue, in order.
2. **Give memory to the running sequences, oldest first.** Keep a count of the
   blocks promised in this step (initially 0); a block is *available* if it is
   free in the engine and not promised. For each running sequence, in order: if it
   needs a new block and none is available, **preempt the most recently admitted
   running sequence** (`engine.preempt`), remove it from the running list, and put
   it at the **front** of the waiting queue; repeat until a block is available or
   the sequence that needs the block has itself been preempted (it was the most
   recent). If it is still running, promise it its block, if it needs one, and it
   decodes in this step.
3. **Admit**, first come first served, **only if step 2 preempted nothing**. While
   fewer than `max_running` sequences are running and the waiting queue is not
   empty, look at its front request. Its prefill is its `prompt_tokens`, or
   `engine.remaining_prefill(id)` if it was preempted, and needs
   $\lceil \text{prefill} / \text{block\_tokens} \rceil$ blocks. It is admitted if
   `available - needed >= watermark_blocks`, **or**, when nothing decodes in this
   step and nothing has been admitted in it yet, if `available - needed >= 0`.
   Admitting promises its blocks, appends it to the running list, and (for a
   request that was never admitted) calls `engine.admit`. If the front request is
   not admitted, admission stops: nobody overtakes it.
4. **Step** once: `prefill` lists the sequences admitted in (3) in that order, each
   with its whole prefill; `decode` lists the sequences that decode by (2), oldest
   first.

When the step returns, finished sequences leave the running list and the next step
is planned at the same instant. With nothing waiting and nothing running, block on
`await queue.get()`. Arrivals at the instant a step ends are in the queue before
the next step is planned, and simultaneous arrivals are queued in order, as in
`continuous-batching-scheduler`.

Two consequences worth knowing: several sequences can be preempted in one step,
and since each goes to the front, they wait in their old running order. And a
preempted sequence never fits back in the step that preempted it, so rule 3's
condition only states what the arithmetic already implies.

## Example

`max_running = 3`, `watermark_blocks = 1`, 7 blocks of 4 tokens. Requests as
(arrival, prompt tokens, output tokens): request 0 `(0, 8, 6)`, request 1
`(0, 8, 4)`, request 2 `(4, 6, 5)`, request 3 `(30, 4, 2)`.

| starts | prefill | decode | blocks held during the step (0, 1, 2, 3) | notes |
|---|---|---|---|---|
| 0 | `(0, 8), (1, 8)` | | 2, 2, -, - | 3 free |
| 12 | | `0, 1` | 3, 3, -, - | both cross a block boundary, 1 block stays available. Request 2 needs 2: $1 - 2 < 1$ |
| 22 | | `0, 1` | 3, 3, -, - | request 2: $1 - 2 < 1$ |
| 32 | | `0, 1` | 3, 3, -, - | request 1 finishes at 42 |
| 42 | `(2, 6), (3, 4)` | `0` | 3, -, 2, 1 | 4 available: request 2 leaves 2, request 3 leaves 1, both at least the watermark |
| 53.5 | | `0, 2` | 4, -, 2, - | request 0 (12 tokens) takes the last block. Request 3 (4 tokens) needs one, none is available, and the most recently admitted sequence is request 3 itself: preempted, after 1 token |
| 63.5 | `(3, 5)` | `2` | -, -, 2, 2 | request 0 finished. Request 3 returns: its prompt and its one token, 5 tokens |
| 73.75 | | `2` | -, -, 3, - | request 3 finished with its second token |
| 82.75 | | `2` | -, -, 3, - | request 2 finishes at 91.75 |

![The worked example, one row per request. Request 3 is preempted at 53.5 ms, waits first in line, and returns at 63.5 ms with a 5-token prefill.](assets/preempt-timeline.svg)

Request 3 produced token 0 at 53.5 and token 1 at 73.75: it was not started over.
The 5 tokens prefilled at 63.5 are the cost of the preemption (*recomputed
tokens*). The watermark of 1 kept request 2 out at 12 and 22, when it would have
taken the last free blocks, but could not prevent the preemption at 53.5, where two
sequences crossed a block boundary in the same step.

## Constraints

- `1 <= max_running <= 16`, `0 <= watermark_blocks < kv_blocks`, block sizes from 1
  to 16 tokens. For every request, `prompt_tokens + output_tokens - 1` tokens fit
  in the pool, so a sequence running alone always finishes. Arrival times are
  multiples of 0.25 ms.
- The policy determines every step and every preemption. The tests compare both
  exactly: with literal schedules on small scenarios and with the reference on
  seeded bursty and Poisson workloads of up to 120 requests. They also check, from
  the trace alone, that every request finishes, that each of its tokens is produced
  exactly once and in order, that no sequence is released or admitted twice, that
  the preempted sequence is always the most recently admitted running one, and
  that at most `max_running` sequences hold memory.
- **Cost.** On a Poisson workload of 120 requests (prompts of 16 to 160 tokens,
  outputs of 8 to 96) with `max_running = 8`, 40 blocks of 16 tokens, and
  `watermark_blocks = 4`, at most **3000** tokens may be recomputed
  (`engine.counters.recomputed_tokens`). The reference recomputes 1759 in 16
  preemptions; without the watermark it is 5527 in 63.
- Use `asyncio` and the engine only.

## Why this matters

An engine cannot know how long an answer will be, so it cannot reserve memory for
it. Reserving the maximum for every sequence wastes most of the KV cache; admitting
optimistically fills the GPU and eventually runs out in the middle of a batch. The
engine then has to choose whom to stop. Stopping the newest sequence wastes the
least work and keeps the service first come, first served; keeping its output and
recomputing only its keys and values makes the preemption invisible to the client
apart from a pause. Get the details wrong and the failure is quiet: tokens
streamed twice, a request that never finishes because it is always the victim, or
an engine that spends its time prefilling the same prompts over and over while
throughput collapses. The count of preemptions is one of the first metrics to look
at when a server is slow under load.
