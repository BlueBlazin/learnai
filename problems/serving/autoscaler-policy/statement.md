Write the function an autoscaler evaluates every few seconds: given what the
cluster looks like now, how many replicas should there be? Adding capacity is the
easy half. The policy is judged by what it does while new replicas are still
starting, and by whether it keeps replicas through a lull that ends ten seconds
later.

## Specification

```python
def desired_replicas(metrics, state, cfg) -> tuple[int, object]: ...
```

The function is called once per evaluation period. It returns `(n, state)`: `n` is
the number of replicas that should be starting or ready from now on, and `state` is
whatever the function wants to be given back at the next evaluation (`None` at the
first). The caller applies `n` at once: it starts replicas, or drains the newest
ones.

**`metrics`**, the cluster at this instant:

| attribute | meaning |
|---|---|
| `now` | the time in seconds |
| `ready` | replicas that are serving |
| `starting` | replicas that were added and are not ready yet |
| `queue_depth` | requests waiting for a slot, over all replicas |
| `running` | requests being served, over all replicas |

**`cfg`**: `target` (an int: how many requests in flight one replica should carry),
`tolerance`, `min_replicas`, `max_replicas`, `window_s`.

**The policy.** Let `current = ready + starting` (at least 1) and
`load = queue_depth + running`, the requests in flight.

1. **Proportional rule.** If `abs(load - current * target) <= tolerance * current *
   target`, the load per replica is close enough to the target and the
   recommendation is `current`. Otherwise it is $\lceil \text{load} /
   \text{target} \rceil$, computed in integers.
2. **Bounds.** The recommendation is raised to `min_replicas` and capped at
   `max_replicas`. Call the result `raw`.
3. **History.** `raw` is recorded with the time `now`. A recorded recommendation is
   *in the window* at a later evaluation at time `t` if it was made at a time `s`
   with `t - s < window_s`. The one just recorded always is.
4. **Scale up at once.** If `raw >= current`, return `n = raw`.
5. **Scale down late.** Otherwise return `n = min(current, highest recommendation in
   the window)`.

So the replica count rises as soon as one evaluation asks for more, and falls only
to the highest level that any evaluation of the last `window_s` seconds asked for.

Step 1 is the Kubernetes rule `desired = ceil(current * metric / target)` with the
metric "requests in flight per replica, averaged over the replicas that are
starting or ready": $\lceil \text{current} \cdot (\text{load} / \text{current}) /
\text{target} \rceil = \lceil \text{load} / \text{target} \rceil$. A replica that
is still starting carries nothing yet and counts as a replica all the same.

**State.** The function must not keep anything between calls except through
`state`, and must not modify the `state` it receives: the tests call it again with
an earlier state and expect the earlier answer. The tests never look inside
`state`.

**In the simulation**, the policy drives the kit's `SimCluster`. A replica runs at
most 8 requests at once (first come, first served; a step costs $2^{-6}$ s plus
$2^{-12}$ s per prefilled token plus $2^{-9}$ s per decoded sequence), so it has
spare slots at `target = 6`. A replica that is added is ready 4 seconds later;
requests go to the ready replica with the fewest requests; a replica that is
removed finishes what it holds first. The policy is evaluated every second from
$t = 0$, with one replica at the start and `cfg = (target 6, tolerance 0.25,
min 1, max 40, window 20 s)`.

## Example

`target = 10`, `tolerance = 0.25`, `min_replicas = 1`, `max_replicas = 8`,
`window_s = 30`, evaluated every 10 seconds.

| `now` | ready | starting | queue | running | recommendation | window | `n` |
|---|---|---|---|---|---|---|---|
| 0 | 2 | 0 | 0 | 14 | $\lceil 14/10 \rceil = 2$ | 2 | 2 |
| 10 | 2 | 0 | 41 | 16 | $\lceil 57/10 \rceil = 6$ | 2, 6 | 6 |
| 20 | 2 | 4 | 36 | 16 | 6 (52 is within 15 of 60) | 2, 6, 6 | 6 |
| 30 | 6 | 0 | 0 | 21 | 3 | 6, 6, 3 | 6 |
| 40 | 6 | 0 | 0 | 18 | 2 | 6, 3, 2 | 6 |
| 50 | 6 | 0 | 0 | 44 | 5 | 3, 2, 5 | 5 |
| 60 | 5 | 0 | 0 | 12 | 2 | 2, 5, 2 | 5 |
| 70 | 5 | 0 | 0 | 9 | 1 | 5, 2, 1 | 5 |
| 80 | 5 | 0 | 0 | 8 | 1 | 2, 1, 1 | 2 |

At 20 the four new replicas are still starting and the backlog is as large as
before. Dividing the load by the two ready replicas would give 26 per replica and
ask for more; dividing by all six gives 8.7, inside the band around 10. At 30 the
recommendation of time 0 is exactly 30 seconds old and has left the window. The
first scale-down happens at 50, when the last recommendation of 6 (made at 20) has
left the window, and it goes to 5, not to the 2 that was recommended in between.

## Constraints

- `1 <= target <= 1000`; `0 <= tolerance < 1`; `1 <= min_replicas <= max_replicas
  <= 1000`; `window_s >= 0`; `ready + starting >= 1`. Evaluation times increase
  strictly but need not be evenly spaced. In the tests `tolerance` is a multiple of
  0.125 and times are multiples of 0.5, so every comparison is exact.
- `n` is a Python `int`. The policy determines it: it is compared exactly, on
  scripts and in the simulation.
- **Load step**: 5 requests per second for 20 s, 30 per second for 40 s, 5 per
  second for 40 s (1478 requests; prompts of 8 to 64 tokens, outputs of 4 to 32),
  simulated for 120 s. **Bursts**: six bursts of 30 requests per second lasting
  6 s, 6 s apart, on a base of 5 per second (1463 requests), simulated for 132 s.
  In both:

  | quantity | limit | reference, step | reference, bursts |
  |---|---|---|---|
  | requests whose first token took more than 1 s (or never came) | 160 step, 250 bursts | 111 | 127 |
  | replica-seconds paid for | 800 step, 900 bursts | 575 | 669 |
  | evaluations that changed the replica count | 18 | 13 | 13 |
  | largest `n` returned | 22 | 15 | 16 |

  The late requests of the reference are those of the first surge, which no policy
  can serve before the first new replica is ready. No request may fail.

## Why this matters

GPU replicas are the most expensive thing in a serving bill and take from tens of
seconds to minutes to start, so an autoscaler is always acting on a cluster that
has not yet reacted to its last decision. A policy that forgets the replicas it
already ordered orders them again at every evaluation and overshoots by a factor;
one that removes replicas as soon as the load dips pays a cold start at every
burst and turns a bursty workload into permanent SLO violations; one that watches
the wrong signal (GPU utilization, or the number of running requests, which are
both pinned at "full" by the time users are queueing) does not see overload at all.
The rule in this problem is the one the Kubernetes Horizontal Pod Autoscaler
applies, fed with the metric LLM servers actually export for this purpose: requests
waiting plus requests running.
