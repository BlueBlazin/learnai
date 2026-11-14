## The idea

A static batcher schedules *requests*: it picks a batch, runs it until every sequence
in it has finished, and only then looks at the queue again. A continuous batcher
schedules *iterations*. Generation already is a loop of forward passes, and nothing
forces two consecutive passes to contain the same sequences. So the scheduler runs
between every two passes and asks a small question: who takes part in the next one?

The answer in this problem is the simplest useful policy:

- every running sequence decodes (nobody is paused);
- every free slot is given to the oldest waiting request, whose prompt is prefilled in
  the same pass that decodes the others;
- a sequence that produced its last token is gone before the next pass is planned.

The state is two collections: `waiting` (a FIFO of requests without a slot) and
`running` (ids in admission order). The engine's contract fixes the rest. A sequence
is prefilled in the step that admits it and decodes from the next step on, so the new
ids must be added to `running` *after* the step. A finished sequence is reported in
`result.finished`, so it must be removed *before* the next admission count.

Why this is correct in the edge cases: arrivals are delivered to the queue before the
step's completion wakes the loop, so draining the queue at the top of each iteration
sees every request that arrived during the step, including at its last instant. And
the idle branch blocks on `queue.get()` only when both collections and the queue are
empty, so a step is never empty and the engine never waits while there is work.

## Reference solution (Python)

```python
import asyncio
from collections import deque


async def run(engine, queue: asyncio.Queue, max_running: int) -> None:
    waiting = deque()  # requests that have arrived and hold no slot yet, oldest first
    running = []       # ids of admitted, unfinished sequences, in admission order
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())      # idle: sleep until an arrival
        while not queue.empty():
            waiting.append(queue.get_nowait())

        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
```

On the test workload (12 requests per second, 8 slots) this loop has a median latency
of about 440 ms against about 1140 ms for static batching, and a p99 time to first
token of about 515 ms against about 1750 ms. Both gaps come from the same place: a
static batch runs as long as its longest member, and for most of that time most of its
slots are empty.

## Common mistakes

- **Waiting for the whole batch.** Guarding admission with "nothing is running" is
  static batching with extra steps. Every trace with an arrival during a step shows
  it, and median latency under load is more than twice as high.
- **Freeing a slot one step late.** Counting free slots before removing the sequences
  that just finished (or returning their slots at the end of the iteration) leaves
  each freed slot empty for one step. The schedule is legal and every request is
  served, which is why only an exact comparison of steps catches it; in production it
  is a few percent of throughput that nobody finds.
- **Decoding a sequence in its prefill step.** Appending new ids to `running` before
  the step lists them in `prefill` and `decode` at once. The prefill pass *is* the
  pass that produces the first token; there is nothing to decode yet. The engine
  refuses the step (`duplicate_sequence`).
- **Losing arrivals.** A loop that admits straight from the queue and has no waiting
  list must put back what does not fit. Dropping it (or overwriting the list on the
  next iteration) loses every request that arrives while the slots are full: those
  clients hang forever.
- **Admitting one request per step.** After a burst, or when several sequences finish
  together, more than one slot is free. `if` instead of `while` fills one of them.
- **Using the waiting list as a stack.** `pop()` instead of `popleft()` serves the
  newest request first; under a backlog the oldest one starves and tail time to first
  token explodes while the median looks fine.
- **Polling when idle.** `await asyncio.sleep(...)` in a loop instead of
  `await queue.get()` starts the next request late by up to the polling interval, and
  burns CPU on an idle server.

## Complexity

Each iteration does $O(R)$ work for $R$ = `max_running` (building the two lists and
filtering `running`), plus $O(1)$ per collected request. A step has one entry per
sequence it advances, so no scheduler can do asymptotically less. Memory is
$O(R + |\text{waiting}|)$. The `deque` matters under a backlog: `list.pop(0)` would
make every admission linear in the length of the waiting list.

## In the wild

- **Orca** (Yu, Jeong, Kim, Kim, and Chun, OSDI 2022) introduced *iteration-level
  scheduling*: the scheduler invokes the engine for a single iteration at a time, so a
  finished request returns at once and a new one starts after at most one iteration.
  Its second idea, *selective batching*, is what makes a mixed step possible at all:
  sequences of different lengths, some in prefill and some in decode, share the batched
  linear layers while attention is computed per sequence. The engine's `prefill` and
  `decode` lists in one call are that mixed step.
- **vLLM**'s scheduler has the same two collections (`waiting`, `running`). Each call
  to `schedule()` first gives every running request its tokens for the step, then
  admits from `waiting` while the budgets allow; the limits are a number of sequences
  (`max_num_seqs`, the slots here) and a number of tokens per step
  (`max_num_batched_tokens`). Finished requests are freed between steps.
- **TGI** exposes the admission rule as `max_batch_total_tokens` and
  `waiting_served_ratio`; **llama.cpp**'s server calls the slots `--parallel` and the
  policy `--cont-batching`; **SGLang** and **TensorRT-LLM** ("in-flight batching")
  schedule per iteration as well.
- What this problem leaves out is what the rest of the track adds: a token budget per
  step and chunked prompts (`chunked-prefill-scheduling`), an order other than first
  come, first served (`priority-scheduler`, `vtc-fair-scheduler`), leaving before the
  end (`deadline-cancellation`), and memory pressure (`preemption-recompute`).
