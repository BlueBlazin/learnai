## The idea

An autoscaler is a feedback controller with a slow actuator. Three decisions make
it work.

**The signal.** Scale on something that keeps growing when the system is
overloaded. Requests in flight (waiting plus running) does: every request the
replicas cannot absorb adds one. Utilization does not. A GPU serving one sequence
and a GPU serving a full batch both report close to 100%, and the number of
*running* requests is capped by the slots of the replicas that exist, so once users
start queueing these signals sit at their ceiling and say nothing about how much
capacity is missing. With a per-replica target $T$ and $L$ requests in flight, the
replica count that restores the target is

$$
\text{desired} = \left\lceil \text{current} \cdot \frac{L / \text{current}}{T} \right\rceil
= \left\lceil \frac{L}{T} \right\rceil .
$$

The left form is how the rule is usually written (current replicas times the ratio
of the measured per-replica value to its target). The right form shows what it
does: it sizes the cluster for the load, in one step, whatever the current size.

**Pending capacity.** The per-replica value must be averaged over every replica
that has been ordered, not only those that are ready. Suppose 2 replicas are ready,
4 more are starting, and the backlog is still 52 because nothing has changed yet.
Averaged over the 2 ready replicas the load is 26 per replica, the ratio to a
target of 10 is 2.6, and the rule asks for $\lceil 6 \cdot 2.6 \rceil = 16$. One
evaluation later it asks for more still. The cluster ends up several times too
large because the policy kept reacting to a problem it had already fixed. Counting
the starting replicas (as carrying nothing yet) gives 8.7 per replica: no action.

**Asymmetric timing.** Being short of capacity costs latency now; having too much
costs money slowly. So scale-up is applied at once and scale-down is delayed: the
count may fall only to the highest recommendation of the last `window_s` seconds.
A lull shorter than the window removes nothing, and a real decline is followed one
window late. Taking the *maximum* of the window, rather than waiting a fixed time
or averaging, has a useful property: every recent evaluation has a veto, so a
single busy second inside a quiet period keeps the capacity it asked for.

**The tolerance band** stops the count from following noise. Without it, a load
that fluctuates by one request around a multiple of the target changes the ceiling
back and forth; with it, nothing happens until the per-replica load is off by more
than the tolerance.

## Reference solution (Python)

```python
def desired_replicas(metrics, state, cfg):
    current = metrics.ready + metrics.starting
    load = metrics.queue_depth + metrics.running

    capacity = current * cfg.target
    if abs(load - capacity) <= cfg.tolerance * capacity:
        raw = current
    else:
        raw = -(-load // cfg.target)
    raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))

    history = [(t, n) for t, n in (state or ()) if metrics.now - t < cfg.window_s]
    history.append((metrics.now, raw))

    if raw >= current:
        n = raw
    else:
        n = min(current, max(n for _, n in history))
    return n, history
```

The state is the window itself. It is rebuilt on every call, so the function is
pure: the same arguments always give the same result, which also makes it trivial
to test and to replay from logs.

## What each rule is worth

The two simulations of the statement (replicas take 4 s to start, the policy runs
every second, a request is late if its first token takes more than 1 s):

| policy | late, step | replica-s, step | late, bursts | replica-s, bursts | actions, bursts | peak |
|---|---|---|---|---|---|---|
| this problem | 111 | 575 | 127 | 669 | 13 | 16 |
| no scale-down window | 188 | 335 | 620 | 649 | 48 | 20 |
| window in both directions | 667 | 1526 | 398 | 494 | 7 | 37 |
| load averaged over ready replicas | 111 | 1165 | 127 | 1255 | 15 | 40 |
| running requests only | 295 | 278 | 127 | 320 | 5 | 4 |
| history never pruned | 111 | 1477 | 127 | 1915 | 4 | 16 |

Without the window, the bursty load costs almost the same replica time and five
times the late requests: the replicas are removed in each gap and are still
starting when the next burst is half over. That is flapping, and its signature is
the action count. Averaging over ready replicas doubles the bill without serving
anyone sooner. Watching only running requests is the cheapest row and, on the load
step, leaves the cluster at 4 replicas with a queue it cannot see.

## Common mistakes

- **No stabilization**, or the same delay in both directions. The first flaps; the
  second answers a load step one window late.
- **Pending replicas ignored**: dividing by `ready`, or computing "how many to add"
  from the backlog at every evaluation without subtracting what is already
  starting.
- **The wrong signal**: utilization or running requests instead of queue plus
  running.
- **Rounding down**: $\lfloor L/T \rfloor$ runs every replica above its target and
  asks for zero replicas under light load.
- **Float ceilings**: `math.ceil(current * (load / current) / target)` can land a
  hair above an integer and round up to the next one. Do the division in integers.
- **A window that never forgets**, or one that keeps an entry exactly `window_s`
  old: the first never scales down, the second is one evaluation late every time.
- **Averaging the window** instead of taking its maximum.
- **Mutating the state** that was passed in. Harmless in a single loop, wrong as
  soon as the policy is replayed, tested, or evaluated speculatively.

## Complexity

One evaluation is $O(w)$ for $w$ recommendations in the window, at most
`window_s / period` of them. A monotonic deque brings the maximum down to amortized
$O(1)$; with a window of a few hundred entries nobody does.

## In the wild

- The **Kubernetes Horizontal Pod Autoscaler** computes `desiredReplicas =
  ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]`, skips scaling
  when the ratio is within a tolerance of 1.0 (0.1 by default), and runs every 15
  seconds by default. Before scaling down it considers all recommendations of a
  stabilization window (5 minutes by default) and uses the highest. Pods that are
  not yet ready are set aside, and when the result would be a scale-up the
  controller assumes they consume 0% of the target, which dampens the scale-up
  exactly as counting `starting` replicas does here (Kubernetes documentation,
  "Horizontal Pod Autoscaling", sections "Algorithm details" and "Stabilization
  window"). Its `behavior` field can also rate-limit each direction, which this
  problem leaves out.
- For LLM servers the usual custom metrics are the engine's own gauges of waiting
  and running requests (vLLM exports `vllm:num_requests_waiting` and
  `vllm:num_requests_running`), supplied to the autoscaler through a metrics
  adapter or an event-driven autoscaler such as KEDA, in place of the CPU or GPU
  utilization that the default configuration scales on.
- This problem's `target` is a concurrency set point: the batch size at which one
  replica still meets its latency objective, less some headroom, which is what
  `capacity-planner` computes.
