## The idea

An autoscaler is a feedback controller driving a slow actuator. Four design choices make it stable and responsive.

**Choosing the right signal.** Scale on a metric that continues growing during overload. Total requests in flight (`queue_depth + running`) grows by one for every request the cluster cannot immediately serve. By contrast, GPU utilization and `running` requests both saturate as soon as queues form. A GPU serving one sequence and a GPU serving a full batch both report near 100% utilization, and `running` is capped by the slots of existing replicas. Once users start queueing, those saturated signals cannot tell the autoscaler how much capacity is missing.

With a per-replica target $T$ and $L$ requests in flight across `current` replicas, the replica count that restores the target load is

$$
\text{desired} = \left\lceil \text{current} \cdot \frac{L / \text{current}}{T} \right\rceil
= \left\lceil \frac{L}{T} \right\rceil .
$$

The left-hand side is the standard proportional formula (current replicas multiplied by the ratio of measured per-replica load to the target). Canceling `current` gives the right-hand side, which sizes the cluster directly for the current load in one step regardless of how many replicas exist today.

**Counting pending capacity.** Per-replica load must be averaged across all ordered replicas (`ready + starting`), not only those already `ready`. Suppose 2 replicas are ready, 4 more are starting, and the backlog is still 52 because the new replicas have not booted yet. Averaging over only the 2 ready replicas yields 26 requests per replica, which is 2.6 times a target of 10 and asks for $\lceil 6 \cdot 2.6 \rceil = 16$ replicas. One evaluation later it asks for even more, overshooting by a large factor because it keeps reacting to a backlog it has already addressed. Counting the 4 starting replicas gives $52 / 6 \approx 8.7$ requests per replica, which stays inside the tolerance band and triggers no extra scaling.

**Asymmetric timing.** Running short of capacity hurts user latency immediately, while holding extra capacity costs money slowly. Scale-up therefore takes effect immediately, while scale-down is damped by a stabilization window. The replica count can drop only to the maximum recommendation recorded over the last `window_s` seconds. Brief lulls shorter than `window_s` remove no replicas, while genuine traffic drops are followed one window later. Taking the *maximum* over the window (rather than an average or a fixed cooldown) gives every recent evaluation a veto, so a single busy second inside a quiet period preserves the capacity it needed.

**The tolerance band.** A deadband prevents the replica count from oscillating on noise. Without it, a load fluctuating by a single request across a multiple of `target` flips the ceiling back and forth. With it, the recommendation stays at `current` until per-replica load drifts outside the tolerance band.

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

The `state` is simply the list of `(time, recommendation)` pairs inside the window. Rebuilding a fresh list on every call keeps `desired_replicas` pure, so identical inputs always produce identical outputs and historical logs can be replayed safely.

## What each rule is worth

The table below compares policy variants on the two simulation workloads from the statement (4 s replica startup, 1 s evaluation period, requests late if time to first token exceeds 1 s):

| policy | late, step | replica-s, step | late, bursts | replica-s, bursts | actions, bursts | peak |
|---|---|---|---|---|---|---|
| this problem | 111 | 575 | 127 | 669 | 13 | 16 |
| no scale-down window | 188 | 335 | 620 | 649 | 48 | 20 |
| window in both directions | 667 | 1526 | 398 | 494 | 7 | 37 |
| load averaged over ready replicas | 111 | 1165 | 127 | 1255 | 15 | 40 |
| running requests only | 295 | 278 | 127 | 320 | 5 | 4 |
| history never pruned | 111 | 1477 | 127 | 1915 | 4 | 16 |

Without a scale-down window, the bursty workload costs nearly the same replica-seconds while causing five times as many late requests. Replicas are torn down during each 6-second gap and are still booting when the next burst is half over. This is classic flapping, visible in the 48 scaling actions. Averaging load over only `ready` replicas doubles the bill without serving requests any faster. Scaling on `running` requests alone is the cheapest row, but on the load step it caps the cluster at 4 replicas while an invisible queue builds up.

## Common mistakes

- **Missing or symmetric stabilization.** Removing the window causes flapping on bursts. Applying the same delay to scale-up leaves the cluster under-provisioned for an entire window during a load step.
- **Ignoring starting replicas.** Dividing load by `ready` instead of `ready + starting`, or repeatedly adding replicas for the same backlog without subtracting pending capacity, causes severe overshoot.
- **Using a saturating signal.** Scaling on GPU utilization or `running` requests misses queued demand.
- **Rounding down.** Using $\lfloor L / T \rfloor$ keeps replicas above their target load and recommends zero replicas under light traffic.
- **Floating-point ceiling errors.** Computing `math.ceil(current * (load / current) / target)` can drift slightly above an exact integer due to floating-point rounding and jump to the next integer. Compute the ceiling in integers with `-(-load // cfg.target)`.
- **Off-by-one or missing window expiry.** Never pruning history prevents the cluster from ever scaling down. Using `<=` instead of `<` retains entries that are exactly `window_s` old and delays every scale-down by one evaluation.
- **Averaging the window.** Averaging recent recommendations instead of taking their maximum drops capacity too early during bursty traffic.
- **Mutating `state` in place.** Modifying the incoming `state` list looks fine in a single forward loop, but breaks replay, speculative evaluation, and unit tests that reuse an earlier state.

## Complexity

Each evaluation scans the $w$ entries currently in the window, taking $O(w)$ time where $w$ is at most `window_s / period`. A monotonic deque could reduce the window maximum to amortized $O(1)$, though with a few hundred entries a plain list scan is simpler and fast enough.

## In the wild

- **Kubernetes Horizontal Pod Autoscaler (HPA)** computes `desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]`, skips scaling when the ratio is within a tolerance of 1.0 (0.1 by default), and evaluates every 15 seconds by default. Before scaling down, it takes the highest recommendation across a stabilization window (5 minutes by default). Pods that are not yet ready are set aside, and during scale-up the controller assumes they consume 0% of the target, which dampens scale-up in the same way counting `starting` replicas does here (Kubernetes documentation, "Horizontal Pod Autoscaling", sections "Algorithm details" and "Stabilization window"). HPA's `behavior` field can also rate-limit scaling in each direction, which this problem omits.
- **LLM serving metrics.** In production, autoscalers typically read engine-level gauges of waiting and running requests (such as vLLM's `vllm:num_requests_waiting` and `vllm:num_requests_running`) via a metrics adapter or KEDA, rather than default CPU or GPU utilization.
- **Choosing `target`.** The `target` parameter is a concurrency set point (the batch size at which one replica meets its latency SLO with headroom), which is what `capacity-planner` computes.
