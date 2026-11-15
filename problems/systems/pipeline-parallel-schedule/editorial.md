## The idea

A pipeline of $S$ stages is $S$ servers in a row, and a sequence is a customer
that walks through all of them once per token and may not start its next lap
before finishing the current one. One sequence alone occupies one server at a
time: per token it needs

$$
C = \sum_i \text{service}_i + \sum_i \text{transfer}_i ,
$$

and during that time every stage but one is idle. The pipeline has room for more:
while stage 1 works on sequence `a`, stage 0 can already work on `b`. With enough
sequences in flight every stage always has something waiting, and then tokens
leave the pipeline at the rate of its slowest stage, one every
$\max_i \text{service}_i$, instead of one every $C$. In the public throughput case
that is 5 ms per token against 17 ms.

The scheduling rule that gets there is the simplest one: a stage never idles when a
job is ready, and it takes the job that has been ready longest. What makes the
problem hard is not the rule but making the timeline *exact*.

**Compute, do not listen.** The tempting design is one coroutine per sequence that
walks through the stages, with a lock per stage. It interleaves correctly and
gets one thing wrong: who gets a stage when two jobs become ready at the same
instant is decided by which coroutine the event loop happens to resume first. A
request that arrives at the moment an older sequence's token comes back takes
stage 0 ahead of it, because its timer was created earlier.

The fix is to notice that nothing in this system is a surprise. When a job starts
at $T$ on stage $i$, three future facts are known immediately: the stage is free
at $T + \text{service}_i$; the job is ready at stage $i + 1$ at
$T + \text{service}_i + \text{transfer}_i$; or, at the last stage, the sequence's
next token is ready at stage 0 at $T + \text{service}_i$. So the scheduler keeps a
**calendar** (a heap of future events) and runs a discrete-event simulation that
happens to be executed in real (simulated) time: sleep until the next event time,
apply *all* events of that instant, and only then let each free stage pick from
its waiting heap, ordered by `(ready time, request index)`. The jobs themselves
are tasks that call `pipe.run` and then `pipe.transfer`; the scheduler never waits
for them, because it already knows when they end.

**Why the schedule is unique.** By induction over event times: the state before
time $T$ determines which stages become free and which jobs become ready at $T$;
the rule then determines which job each free stage starts at $T$ (a total order
on waiting jobs); and those starts determine the future events. No choice is left
anywhere.

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

A task created at time `now` starts running when the scheduler next awaits, still
at `now`, so `pipe.run` is called at the planned instant. The job task calls
`pipe.transfer` in the same breath as `pipe.run` returns, which is the instant the
stage finishes. All times are integers, so the calendar and the clock agree
exactly.

An online design also works (one worker per stage that waits for its heap to be
non-empty), provided each worker lets the other events of the instant arrive
before it pops. That is the same "events first, decisions second" order, obtained
by yielding to the event loop instead of by reading a calendar.

## Common mistakes

- **One sequence at a time.** Looping over the requests and walking each through
  the stages is correct and uses one device's worth of the pipeline: 30 ms
  instead of 18 in the worked example, 4080 ms instead of 1212 in the throughput
  case.
- **One token in flight.** A global lock around "one token through all stages"
  lets sequences take turns, and still only one stage works at any time.
- **Letting a sequence's next token in early.** Token $t + 1$ depends on token
  $t$'s output. Starting it when token $t$ has merely left stage 0 would be
  pipelining within a sequence, which autoregressive decoding does not allow.
- **Free transfers.** Marking a job ready at the next stage the moment the
  previous stage finishes starts that stage on an activation that is still in
  transit.
- **Transfers started late.** Claiming the next stage first and transferring
  then (pull instead of push) keeps the receiving stage idle for the whole
  transfer and delays everything behind it. The activation should already be
  there when the stage frees up.
- **A lock per stage.** Ties at one instant are decided by the event loop's
  wake-up order. See above.
- **Priority by request instead of by ready time.** Always serving the earliest
  request lets its tokens overtake jobs that have waited longer, and the later
  requests starve.
- **Ties the wrong way round**, new arrivals before the returning token of an
  older sequence.

## Complexity

One calendar event per stage visit and two heap operations per event:
$O(N S \log(N S))$ scheduling work for $N$ tokens, and $N S$ tasks. The makespan
is at least $N \cdot \max_i \text{service}_i$ (every token passes the slowest
stage) and at least $C$ times the longest sequence; with enough concurrent
sequences the first bound is what the schedule approaches.

## In the wild

- vLLM combines the two kinds of model parallelism: its documentation on
  parallelism and scaling describes setting `tensor_parallel_size` to the number
  of GPUs per node and `pipeline_parallel_size` to the number of nodes, so the
  slow links between nodes carry one activation per token instead of an
  all-reduce per layer.
- The Megatron-LM paper (arXiv 1909.08053) describes its tensor-parallel scheme
  as "orthogonal and complementary" to pipeline model parallelism.
- A device map that spills layers from GPU to CPU is a pipeline with one very slow
  stage. The bound $N \cdot \max_i \text{service}_i$ says what that costs: the
  slow stage sets the throughput no matter how fast the GPU part is.
- Real engines also batch: a stage processes the waiting jobs of several
  sequences in one forward pass. This problem keeps one job per stage at a time to
  isolate the scheduling; batching changes the service time, not the structure.
