## The idea

Think of an $S$-stage pipeline as $S$ servers in series. Each sequence walks through all $S$ stages once per token and cannot start its next token until the current one exits stage $S - 1$. A single sequence occupies only one server at a time, taking

$$
C = \sum_i \text{service}_i + \sum_i \text{transfer}_i
$$

milliseconds per token while all other stages sit idle. Running multiple sequences concurrently fills the pipeline. While stage 1 processes sequence `a`, stage 0 can already process `b`. With enough sequences in flight, every stage stays busy and tokens exit at the rate of the slowest stage, one every $\max_i \text{service}_i$ ms instead of one every $C$ ms. In the public throughput test, that reduces the per-token interval from 17 ms to 5 ms.

The scheduling policy itself is simple. A stage never idles when a job is ready, and it picks the job that has been ready longest. The hard part is making the event timeline exact.

**Compute the timeline instead of racing coroutines.** A natural first attempt is one coroutine per sequence walking through the stages with an `asyncio.Lock` per stage. That interleaves sequences, but it breaks tie-breaking. When two jobs become ready at the same instant, whichever coroutine the event loop wakes first acquires the lock. For example, a new request whose `sleep` timer was created at $t = 0$ can wake ahead of an older sequence's returning token at the same timestamp.

The clean solution relies on the fact that every duration is deterministic. When a job starts at time $T$ on stage $i$, we immediately know its future events:

- Stage $i$ becomes free at $T + \text{service}_i$.
- If $i < S - 1$, the job becomes ready at stage $i + 1$ at $T + \text{service}_i + \text{transfer}_i$.
- If $i = S - 1$, the sequence's next token (if any) becomes ready at stage 0 at $T + \text{service}_i$.

The scheduler therefore maintains a **calendar** (a min-heap of future events) and runs a discrete-event simulation synchronized to the event loop clock. At each step, it sleeps until the next event timestamp, processes *all* events at that timestamp, and only then assigns ready jobs to free stages from each stage's waiting heap ordered by `(ready_time, request_index)`. Each started job runs in its own task that calls `pipe.run` followed immediately by `pipe.transfer`. The scheduler does not need to await individual jobs because it already knows their exact finish times.

**Why the schedule is unique.** By induction over event timestamps, the state before time $T$ uniquely determines which stages become free and which jobs become ready at $T$. Because `(ready_time, request_index)` defines a total order on waiting jobs, the selection rule uniquely determines which job each free stage starts at $T$, which in turn determines all future events.

## Reference solution (Python)

```python
READY, FREE = 0, 1


async def run_pipeline(pipe, requests):
    n_stages = pipe.n_stages
    service, transfer = pipe.service_ms, pipe.transfer_ms
    loop = asyncio.get_running_loop()
    outputs = {seq: [0] * n_tokens for seq, _, n_tokens in requests}

    calendar = [(arrival, READY, 0, index, 0) for index, (_, arrival, _) in enumerate(requests)]
    heapq.heapify(calendar)
    waiting = [[] for _ in range(n_stages)]
    free = [True] * n_stages
    jobs = []

    async def job(stage, index, token):
        seq = requests[index][0]
        result = await pipe.run(stage, seq, token)
        if stage < n_stages - 1:
            await pipe.transfer(stage, seq, token)
        else:
            outputs[seq][token] = result

    while calendar:
        now = calendar[0][0]
        await asyncio.sleep(now - loop.time())

        while calendar and calendar[0][0] == now:
            _, kind, stage, index, token = heapq.heappop(calendar)
            if kind == FREE:
                free[stage] = True
            else:
                heapq.heappush(waiting[stage], (now, index, token))

        for stage in range(n_stages):
            if not (free[stage] and waiting[stage]):
                continue
            _, index, token = heapq.heappop(waiting[stage])
            free[stage] = False
            jobs.append(asyncio.create_task(job(stage, index, token)))
            done = now + service[stage]
            heapq.heappush(calendar, (done, FREE, stage, -1, -1))
            if stage < n_stages - 1:
                arrives = done + transfer[stage]
                heapq.heappush(calendar, (arrives, READY, stage + 1, index, token))
            elif token + 1 < requests[index][2]:
                heapq.heappush(calendar, (done, READY, 0, index, token + 1))

    await asyncio.gather(*jobs)
    return outputs
```

A task created at time `now` starts executing on the scheduler's next `await` while the clock is still at `now`, so `pipe.run` begins at the exact planned timestamp. Inside `job`, `pipe.transfer` starts immediately after `pipe.run` returns, matching the exact finish time of the stage. Because all durations are integers, the calendar and event loop clock stay exactly aligned.

An online design with one worker per stage also works, provided each worker yields until all events at the current timestamp have been processed before popping from its heap. That achieves the same "events first, decisions second" ordering through event-loop yields rather than an explicit calendar.

## Common mistakes

- **Running one sequence at a time.** Looping over requests and finishing each completely before starting the next uses only one device's worth of capacity. That takes 30 ms instead of 18 ms in the worked example, and 4080 ms instead of 1212 ms in the throughput case.
- **Allowing only one token in flight.** Putting a global lock around a token's full trip through all stages lets sequences alternate, but still keeps only one stage busy at a time.
- **Starting a sequence's next token early.** Token $t + 1$ depends on token $t$'s generated output from the last stage. Launching token $t + 1$ as soon as token $t$ leaves stage 0 violates autoregressive data dependence.
- **Treating transfers as zero-cost.** Marking a job ready at stage $i + 1$ as soon as stage $i$ finishes starts stage $i + 1$ before the activation arrives and raises `PipelineError`.
- **Starting transfers late.** Acquiring stage $i + 1$ before running `pipe.transfer` (pull instead of push) leaves stage $i + 1$ idle during the transfer and delays every job behind it. Transfers should overlap with computation on stage $i + 1$.
- **Using an `asyncio.Lock` per stage.** Simultaneous arrivals and completions are then ordered by the event loop's coroutine wake-up order rather than by `(ready_time, request_index)`.
- **Ordering by request index instead of ready time.** Always prioritizing the earliest request lets its newer tokens jump ahead of jobs that have been waiting longer, starving later requests.
- **Breaking ties in the wrong direction.** Preferring a newly arriving request over the returning token of an older sequence flips the request-index tie-breaker.

## Complexity

Let $N$ be the total number of generated tokens across all requests. Each token visits $S$ stages, creating $O(N S)$ calendar events and $N S$ tasks. With $O(\log(N S))$ heap operations per event, scheduling overhead is $O(N S \log(N S))$. The makespan is bounded below by $N \cdot \max_i \text{service}_i$ (every token must pass through the bottleneck stage) and by $C$ times the longest sequence's token count. With enough concurrent sequences, the schedule approaches the bottleneck bound.

## In the wild

- vLLM combines tensor and pipeline parallelism in multi-node deployments. Its scaling documentation recommends setting `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes, so slower cross-node links carry one activation per token instead of an all-reduce at every layer.
- The Megatron-LM paper (arXiv 1909.08053) notes that its tensor-parallel scheme is orthogonal and complementary to pipeline model parallelism.
- A device map that offloads some layers from GPU to CPU creates a pipeline with one very slow stage. The $N \cdot \max_i \text{service}_i$ lower bound shows why offloading hurts throughput regardless of how fast the GPU stages are.
- Production serving engines also batch multiple waiting jobs into a single forward pass at each stage. This problem limits each stage to one job at a time to isolate the scheduling mechanics, while batching changes the per-step service time without changing the pipeline structure.
