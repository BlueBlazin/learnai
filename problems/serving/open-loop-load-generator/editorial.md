## The idea

A load generator models how users send traffic, and there are two standard models.

In a **closed loop**, a fixed pool of virtual users each send a request, wait for the response, and then send the next one. The arrival rate is an output equal to the number of users divided by the response time, so the load drops whenever the server slows down. In an **open loop**, requests arrive on a schedule that ignores server behavior. The arrival rate is an input, and when the server slows down, requests pile up just as they do in production.

The Poisson process is the standard open-loop arrival schedule. Its inter-arrival gaps are independent exponential random variables with mean $1 / \text{rate}$. This matches the aggregate traffic of many independent users and includes natural bursts followed by lulls, which expose queuing behavior that constant-rate schedules miss. To draw an exponential gap $g$ from a uniform $u \in [0, 1)$, invert the exponential CDF $1 - e^{-\text{rate}\cdot g} = u$ to get $g = -\ln(1 - u) / \text{rate}$. Using $1 - u$ instead of $u$ guarantees that the argument to $\ln$ is in $(0, 1]$, so it is never 0.

Two implementation rules make the generator truly open-loop:

1. **The pacing loop awaits only the clock.** Each request runs in its own task. The pacing loop sleeps until the next *absolute* intended send time so that small scheduling delays never accumulate.
2. **Latency is `done_s - intended_s`.** When a request waits for a `max_in_flight` slot, that wait is part of the latency its user experiences. Measuring from `sent_s` instead erases the queuing delay caused by overload.

With `max_in_flight=1`, requests execute one at a time like a closed loop, and the worked example shows why the second rule still matters. Measuring from `intended_s` reports three 1.0 s requests, whereas measuring from `sent_s` reports only one.

## Reference solution (Python)

```python
async def generate_load(send, rate, n, uniforms, max_in_flight=None):
    loop = asyncio.get_running_loop()
    slots = None if max_in_flight is None else asyncio.Semaphore(max_in_flight)
    records = [None] * n

    async def one(i, intended):
        if slots is not None:
            await slots.acquire()
        sent = loop.time()
        try:
            await send(i)
            ok = True
        except OSError:
            ok = False
        finally:
            if slots is not None:
                slots.release()
        done = loop.time()
        records[i] = {
            "intended_s": intended,
            "sent_s": sent,
            "done_s": done,
            "latency_s": done - intended,
            "ok": ok,
        }

    tasks = []
    intended = loop.time()
    for i in range(n):
        intended += -math.log(1.0 - uniforms[i]) / rate
        delay = intended - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(loop.create_task(one(i, intended)))
    if tasks:
        await asyncio.wait(tasks)
    return records
```

`asyncio.Semaphore` wakes waiters in FIFO order, and tasks are created in request order, so waiting requests are sent first come, first served.

## Common mistakes

- **Building a closed loop.** Calling `await send(i)` inside the pacing loop serializes traffic. Against a server that takes 2 s per response with 0.25 s gaps, requests go out every 2 s, applying only one-eighth of the intended load.
- **Measuring latency from `sent_s`.** A request that waits 0.9 s for a slot and completes in 0.1 s gets recorded as 0.1 s. In a test where one response stalls for 5 s on a single slot, 19 requests fall due during the stall and all 19 report 10 ms if measured from `sent_s`.
- **Letting the schedule slip.** Acquiring a semaphore slot inside the pacing loop before sleeping for the next gap pushes all later intended times forward. The schedule then depends on response times, turning the generator back into a closed loop.
- **Using uniform or constant gaps.** The average rate is right, but traffic is too smooth. Without bursts, queues stay artificially short and tail latency looks better than it is.
- **Starting the schedule at time 0.** Computing absolute times from `0` instead of `loop.time()` breaks whenever the generator starts after a warm-up or earlier stage. All initial requests appear overdue and fire in one burst.
- **Letting one `OSError` abort the run.** An uncaught exception in a request task cancels `gather` and discards all other measurements. Under overload, failures are expected and must be recorded with `"ok": False` so metrics like `goodput-slo` count them in the denominator.
- **Returning before all tasks finish.** Failing to await the spawned tasks omits slow, in-flight requests from the returned records, introducing survivorship bias.
- **Ignoring `max_in_flight` or silently dropping requests.** Dropping requests hides overload unless recorded explicitly. Here, excess requests wait for a slot and pay for that wait in their latency.

## Complexity

Time and memory are both $O(n)$. At most `max_in_flight` requests (or, when unbounded, as many as the server's latency allows to overlap) are in flight simultaneously. In production load generators, an unbounded open loop against a stalled server can accumulate requests until it exhausts available sockets.

## In the wild

- **wrk2** (`giltene/wrk2`) popularized the latency correction for coordinated omission. It takes a target throughput rate and measures response latency from the time the request should have been transmitted. Its `--u_latency` flag prints the uncorrected histogram beside the corrected one to highlight the difference. Its README explains how closed-loop generators coordinate with a stalled server and avoid sampling during slow periods.
- **vLLM's benchmark** (`vllm bench serve`) exposes `--request-rate` (requests per second, defaulting to infinity so all requests fire at time 0), `--burstiness` (defaulting to `1.0` for a Poisson process), and `--max-concurrency` (the equivalent of `max_in_flight`). When `--max-concurrency` is too small for the requested rate, the achieved rate falls below the target, matching the second example table. Its `--goodput` option evaluates the SLOs from `goodput-slo`.
- The statistical test uses a bounded transform because Hoeffding's inequality requires bounded random variables and exponential gaps are unbounded. The transform $e^{-\text{rate}\cdot g}$ maps each gap into $(0, 1]$, and for an exponential gap with parameter $\text{rate}$ it is uniform on $(0, 1)$ with mean $\int_0^1 x\,dx = \tfrac12$.
