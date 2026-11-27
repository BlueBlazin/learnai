Implement an asynchronous scheduler that runs multiple autoregressive generation requests through a multi-stage model pipeline. When a model is split into consecutive blocks of layers across devices (as in `device-map-solver`), running one sequence at a time leaves every other device idle. Your scheduler keeps stages busy with different sequences simultaneously, deciding at every millisecond which waiting token each free stage computes next under deterministic rules.

## Specification

```python
async def run_pipeline(pipe, requests: list[tuple[str, int, int]]) -> dict[str, list[int]]: ...
```

**Time.** Time is measured by the `asyncio` event loop clock, where 1 unit equals 1 millisecond (`asyncio.sleep(3)` waits 3 ms). The clock starts at 0 when `run_pipeline` is called. All durations and arrival times are integers. Use `asyncio` time only, never the system wall clock.

**The requests.** Each entry in `requests` is a tuple `(seq_id, arrival_ms, n_tokens)` representing a sequence with a unique string `seq_id` that arrives at `arrival_ms` and requires `n_tokens` $\ge 1$ generated tokens. `requests` is sorted by `arrival_ms`. A request's 0-based position in `requests` is its **request index**.

**The pipeline.** `pipe` has $S =$ `pipe.n_stages` stages, numbered $0, \dots, S - 1$ in layer execution order. To generate token $t$ of sequence $s$, every stage $0, \dots, S - 1$ must process it in order. We refer to the pair $(s, t)$ as a **job**.

- `await pipe.run(stage, seq_id, token)` runs `stage` on that job for `pipe.service_ms[stage]` ms (a positive integer). The last stage ($S - 1$) returns the generated token id (`int`), and every earlier stage returns `None`.
- `await pipe.transfer(stage, seq_id, token)` transfers the output of `stage` for that job to `stage + 1`. It takes `pipe.transfer_ms[stage]` ms (a positive integer; `pipe.transfer_ms` has $S - 1$ entries). Any number of transfers may be in flight at once.

The simulated hardware enforces four **physical rules**, and violating any of them raises `PipelineError`:

1. A stage computes at most one job at a time. A new `run` on a stage may start only after that stage's previous `run` has finished.
2. Stage $i > 0$ may start a job only after that job's transfer from stage $i - 1$ has finished. A transfer may start only after stage $i - 1$ has finished the job.
3. Stage 0 may start token 0 of a sequence only at or after its `arrival_ms`.
4. Stage 0 may start token $t > 0$ of a sequence only after token $t - 1$ has finished the **last** stage ($S - 1$), because autoregressive decoding feeds the previous token back into stage 0. Returning that token id to stage 0 takes 0 ms.

An operation that starts at time $T$ and takes $\delta$ ms is *finished* at time $T + \delta$, and any operation waiting on it may start at exactly $T + \delta$. The hardware checks these rules against timestamps rather than coroutine return order.

**The schedule.** `run_pipeline` must produce this exact schedule:

- A job's **ready time** at a stage is the earliest moment rules 2 through 4 allow it to start there. At stage 0, this is `arrival_ms` for token 0, or the finish time of token $t - 1$ at the last stage for token $t > 0$. At stage $i > 0$, this is the finish time of its transfer from stage $i - 1$. **Every transfer must start the instant stage $i - 1$ finishes the job**, so the ready time at stage $i$ is that finish time plus `transfer_ms[i - 1]`.
- **No stage idles while a job is ready for it.** Whenever a stage is free and at least one job is ready for it (ready time $\le$ current time), the stage starts a job immediately.
- **Selection order.** Among ready jobs for a free stage, pick the one with the earliest ready time. Break ties in ready time by choosing the job with the smaller request index.

Every decision at time $T$ must account for everything that happens at $T$. A job that becomes ready at $T$ competes for a stage that becomes free at $T$, as well as against every other job ready at or before $T$, regardless of the order in which your code observes those events.

**The result.** Return once every token of every request has finished the last stage. The return value is a dict mapping each `seq_id` to the list of its generated token ids (returned by the last stage) in token order.

## Example

Consider two stages with `service_ms = (2, 3)`, `transfer_ms = (1,)`, and `requests = [("a", 0, 2), ("b", 0, 2), ("c", 4, 1)]`. Write token $t$ of sequence `a` as `a[t]`:

| time | stage 0 | stage 1 | notes |
|---|---|---|---|
| 0 | starts `a[0]` | | `a[0]` and `b[0]` are both ready at 0, and `a` has the smaller request index |
| 2 | starts `b[0]` | | `a[0]` starts transferring to stage 1 and arrives at 3 |
| 3 | | starts `a[0]` | |
| 4 | starts `c[0]` | | `c` arrives as stage 0 frees up; `b[0]` arrives at stage 1 at 5 and waits |
| 6 | starts `a[1]` | starts `b[0]` | `a[0]` finishes at 6, so `a[1]` is ready at 6 when stage 0 becomes free |
| 9 | starts `b[1]` | starts `c[0]` | `c[0]` has been ready since 7, whereas `a[1]` became ready at 9 |
| 12 | | starts `a[1]` | `b[1]` also arrives at 12, but `a[1]` has been ready since 9 |
| 15 | | starts `b[1]` | |
| 18 | | | done |

![The example's five tokens on two stages. Each transfer takes 1 ms, and a token waits when stage 1 is busy.](assets/pipeline-timeline.svg)

All five tokens finish by 18 ms. Running one token in the pipeline at a time takes $2 + 1 + 3 = 6$ ms per token, or 30 ms in total.

As a tie-breaking example, take `service_ms = (2, 2)`, `transfer_ms = (1,)`, and `requests = [("a", 0, 2), ("b", 5, 1)]`. Job `a[0]` runs on stage 0 until 2 ms and on stage 1 from 3 to 5 ms. At time 5, two jobs become ready for stage 0 simultaneously: `a[1]` (because `a[0]` just finished the last stage) and `b[0]` (because `b` arrives). Since `a` has the smaller request index, stage 0 starts `a[1]` at 5 ms and `b[0]` at 7 ms.

## Constraints

- $1 \le S \le 8$, with integer service and transfer times from 1 to 20. There are at most 16 requests of at most 40 tokens each, and arrival times are integers $\ge 0$. All timestamps are therefore exact integers.
- The tests record the start time of every `run` and `transfer` call and compare the set of `(time, stage, sequence, token)` starts against the unique schedule defined above. Returned token lists are compared for exact equality.
- The time at which `run_pipeline` returns (the makespan) must equal that schedule's makespan. One public test also checks a reference-independent bound: with 8 sequences of 30 tokens on 4 stages, throughput must be within 5 percent of the bottleneck stage's rate, meaning the makespan is at most `240 * max(service_ms) / 0.95` ms.
- Raising `PipelineError` fails the test case even if your code catches it.

## Why this matters

Pipeline parallelism distributes a model across devices when tensor parallelism is impractical, such as across nodes with slow interconnects (vLLM's `--pipeline-parallel-size`), across unequal GPUs, or across GPU and CPU with a device map. With only one sequence in flight, every device waits on the others and $S$ devices deliver the throughput of one.

Pipeline utilization recovers only when stages stay busy on *different* sequences concurrently. In that regime, the slowest stage sets throughput rather than the sum of all stage times. A scheduler that serializes sequences, delays transfers, or leaves a stage idle while an activation is ready forfeits the throughput that the extra devices were meant to provide.
