## The idea

A load generator models users. There are two models.

In a **closed loop**, a fixed number of virtual users each send a request, wait for
the answer, and send the next. The arrival rate is an output: it equals the number
of users divided by the response time, so when the server slows down, the load
falls. In an **open loop**, requests arrive according to a schedule that does not
look at the server. The rate is an input, and when the server slows down, requests
pile up, as they do in production, where the next user does not know that the
previous one is still waiting.

The Poisson process is the standard open-loop schedule: independent exponential
gaps with mean $1/\text{rate}$. It is what many independent users produce in
aggregate, and it has the bursts (several arrivals close together, then a lull)
that a constant-rate schedule lacks and that queues are sensitive to. A gap is
drawn from a uniform $u$ by inverting the exponential distribution function:
$1 - e^{-\text{rate}\cdot g} = u$ gives $g = -\ln(1 - u)/\text{rate}$. Using $1 - u$
rather than $u$ matters only at the edge: $u$ is in $[0, 1)$, so $1 - u$ is never 0
and the logarithm is always finite.

Two things make the implementation open:

1. **The pacing loop awaits nothing but the clock.** Each request runs in its own
   task. The loop sleeps until the next *absolute* intended time, so even its own
   lateness does not accumulate.
2. **Latency is `done - intended`.** If a request could not be sent when it was due
   (here: because `max_in_flight` connections were busy), the time it waited is part
   of what its user experienced. Subtracting the actual send time removes exactly
   the part of the latency that the overload caused.

With `max_in_flight=1` the generator has the structure of a closed loop, one request
at a time, and the worked example shows the difference the second point makes: the
same run reports one slow request or three.

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

`asyncio.Semaphore` wakes its waiters in the order they started waiting, and the
tasks are created in request order, which gives first come, first served.

## Common mistakes

- **A closed loop.** `await send(i)` in the pacing loop. With a server that answers
  in 2 s and gaps of 0.25 s, the requests go out every 2 s: an eighth of the load
  the test claims to apply.
- **Latency from the send time.** The record of a request that waited 0.9 s for a
  connection and was answered in 0.1 s says 0.1 s. In the test where one response
  stalls for 5 s on a single connection, 19 further requests fall due during the
  stall; measured from the send time, all 19 report 10 ms.
- **Letting the schedule slip.** Taking the connection inside the pacing loop and
  then sleeping for the next gap: every wait pushes all later arrivals back. The
  intended times are now a function of the responses, which is a closed loop again
  with extra steps.
- **Uniform (or constant) gaps.** The mean rate is right and the process is too
  regular: no two requests arrive close together, queues stay shorter than under
  real traffic, and tail latency looks better than it is.
- **Starting the schedule at clock zero.** Absolute times computed from 0 instead of
  from the call: if the generator starts later (after a warm-up, as the second stage
  of a sweep) the first requests are all overdue and go out as one burst.
- **One failure ends the run.** An uncaught exception in a task surfaces in
  `gather` and takes the measurements of all other requests with it. Under overload
  failures are the expected outcome; `goodput-slo` needs them in the denominator.
- **Returning before the responses.** Forgetting to wait for the tasks returns
  records for the fast requests only, which is survivorship bias in one line.
- **Ignoring the limit,** or enforcing it by dropping requests silently. If
  requests are dropped, the result must say so; here they wait and are charged for
  it.

## Complexity

$O(n)$ time and memory; at most `max_in_flight` requests (or, without a limit, as
many as the server's latency allows) are in flight at once. In a real generator the
unbounded case needs care: an open loop against a stalled server accumulates
requests, which is the point, and also a way to run out of sockets.

## In the wild

- **wrk2** (the README of `giltene/wrk2`) is the reference for the latency half. It
  takes a constant-throughput rate argument, and in its README's words "wrk2
  measures response latency from the time the transmission should have occurred";
  `--u_latency` prints the uncorrected histogram next to the corrected one to show
  the size of the effect. The README also describes coordinated omission in the
  tool it derives from: each connection sends a request only after receiving a
  response, so high-latency responses make the load generator coordinate with the
  server and avoid measuring during the slow periods.
- **vLLM's benchmark** (`vllm bench serve`) has `--request-rate` (requests per
  second; with the default, infinity, all requests are sent at time 0),
  `--burstiness` (the default 1.0 is a Poisson process), and `--max-concurrency`,
  the `max_in_flight` of this problem. With a concurrency limit below what the rate
  needs, the achieved rate falls short of the requested one, which is the situation
  of the second example table. Its `--goodput` option takes the objectives of
  `goodput-slo`.
- The statistical check uses a bounded transform because Hoeffding's inequality
  needs bounded values and a gap is unbounded: $e^{-\text{rate}\cdot g}$ maps the gap
  to $(0, 1]$, and for an exponential gap it is uniform, with mean
  $\int_0^1 x\,dx = \tfrac12$.
