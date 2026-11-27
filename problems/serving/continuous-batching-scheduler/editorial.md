## The idea

A static batcher schedules *requests*. It picks a batch, runs it until every sequence finishes, and only then checks the queue again. A continuous batcher schedules *iterations*. Because generation is already a loop of forward passes, consecutive passes do not need to contain the same sequences. Between every two passes, the scheduler decides which sequences take part in the next one.

This problem implements the simplest useful continuous-batching policy:

- Every running sequence decodes on every step without being paused.
- Every free slot goes to the oldest waiting request, whose prompt is prefilled in the same step that decodes existing sequences.
- Any sequence that produces its last token leaves before the next step is planned.

You only need two data structures: `waiting` (a FIFO queue of requests without a slot) and `running` (a list of live sequence ids in admission order). The engine's contract fixes the ordering of updates. Because a newly admitted sequence is prefilled in its first step and decodes starting on the next step, new ids must join `running` *after* the step completes. Likewise, finished sequences from `result.finished` must be removed *before* counting free slots for the next step.

This structure also handles the timing edge cases cleanly. Arrivals are pushed into `queue` before step completion resumes the loop, so draining `queue` at the top of each iteration captures every request that arrived during the step, including at its final instant. Blocking on `queue.get()` only when `waiting`, `running`, and `queue` are all empty ensures you never submit an empty step and never leave the engine idle when work exists.

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

On the benchmark workload (12 requests per second, 8 slots), this loop achieves a median end-to-end latency of about 440 ms compared to about 1140 ms for static batching, and a p99 time to first token of about 515 ms compared to about 1750 ms. Both gains happen because a static batch lasts as long as its longest sequence, leaving most slots empty for most of that time.

## Common mistakes

- **Waiting for the whole batch to finish.** Guarding admission with `if not running:` turns the loop back into static batching. Any trace with arrivals during a step exposes this bug, and median latency under load more than doubles.
- **Freeing a slot one step late.** Counting free slots before removing `result.finished` leaves each freed slot empty for an extra step. Every request still finishes legally, so only exact step comparison catches the bug, yet in production it silently wastes throughput.
- **Decoding a sequence in its prefill step.** Appending newly admitted ids to `running` before calling `engine.step` puts them in both `prefill` and `decode`. The prefill pass already produces the first token, so the engine rejects the step with `duplicate_sequence`.
- **Losing arrivals.** Pulling straight from `queue` without a `waiting` buffer—or overwriting `waiting` on each iteration—drops requests that arrive while all slots are full, causing those clients to hang forever.
- **Admitting at most one request per step.** After a burst or when multiple sequences finish together, several slots are free at once. Using `if` instead of `while` leaves the extra slots unused until later steps.
- **Treating `waiting` as a stack.** Calling `pop()` instead of `popleft()` serves the newest request first. Under a backlog, older requests starve and tail time to first token explodes even if the median looks fine.
- **Polling when idle.** Calling `await asyncio.sleep(...)` in a loop instead of `await queue.get()` delays new requests by up to the polling interval and burns CPU on an idle server.

## Complexity

Let $R = \text{max\_running}$. Each iteration does $O(R)$ work (building `prefill` and `decode` and filtering `running`), plus $O(1)$ work per collected request. Because a step lists every sequence it advances, no scheduler can do asymptotically less. Memory usage is $O(R + |\text{waiting}|)$. Using a `deque` matters under a backlog, since `list.pop(0)` would make each admission linear in the length of `waiting`.

## In the wild

- **Orca** (Yu, Jeong, Kim, Kim, and Chun, OSDI 2022) introduced *iteration-level scheduling*, where the scheduler invokes the engine for one iteration at a time so finished requests return immediately and new ones start after at most one step. Its companion technique, *selective batching*, enables mixed steps by batching linear layers across sequences of different lengths while computing attention per sequence. Passing both `prefill` and `decode` into one `engine.step` call models that mixed pass.
- **vLLM** maintains the same `waiting` and `running` collections. On each `schedule()` call, it allocates tokens for running requests first and then admits waiting requests while staying within both the sequence limit (`max_num_seqs`, corresponding to `max_running` here) and the per-step token budget (`max_num_batched_tokens`). Finished requests are freed between steps.
- **Other serving engines** use the same pattern under different flags. TGI exposes `max_batch_total_tokens` and `waiting_served_ratio`, llama.cpp's server uses `--parallel` and `--cont-batching`, and both SGLang and TensorRT-LLM ("in-flight batching") schedule at iteration granularity.
- **Beyond this problem.** Later problems in this track extend this loop with per-step token budgets and chunked prompts (`chunked-prefill-scheduling`), non-FCFS ordering (`priority-scheduler`, `vtc-fair-scheduler`), early exits (`deadline-cancellation`), and memory pressure (`preemption-recompute`).
