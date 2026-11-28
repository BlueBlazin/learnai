Implement `desired_replicas`, the policy function an autoscaler evaluates every few seconds to decide how many replicas the cluster should have. Adding capacity during a traffic spike is straightforward, so the policy is judged on two harder behaviors. It must account for replicas that are still starting up, and it must hold onto replicas through a brief lull that ends seconds later.

![This problem is the Autoscaler, which sets the number of replicas from the metrics.](assets/map.svg)

## Specification

```python
def desired_replicas(metrics, state, cfg) -> tuple[int, object]: ...
```

The autoscaler calls `desired_replicas` once per evaluation period. It returns `(n, state)`, where `n` is the target number of replicas (starting or ready) as a Python `int`, and `state` is whatever value the function wants passed back on the next evaluation (`None` on the first call). The caller applies `n` immediately by starting new replicas or draining the newest ones.

**`metrics`** describes the cluster at the current instant:

| attribute | meaning |
|---|---|
| `now` | current time in seconds |
| `ready` | replicas currently serving requests |
| `starting` | replicas that were added and are not ready yet |
| `queue_depth` | requests waiting for a slot across all replicas |
| `running` | requests currently being served across all replicas |

**`cfg`** holds the policy parameters:

| attribute | meaning |
|---|---|
| `target` | integer target number of in-flight requests per replica |
| `tolerance` | deadband fraction around `target` within which no scaling occurs |
| `min_replicas` | minimum allowed replica count |
| `max_replicas` | maximum allowed replica count |
| `window_s` | scale-down stabilization window in seconds |

**The policy.** Let `current = ready + starting` (at least 1) and `load = queue_depth + running` (the total requests in flight).

1. **Proportional rule.** If `abs(load - current * target) <= tolerance * current * target`, the load per replica is close enough to `target` and the recommendation is `current`. Otherwise, compute the recommendation as $\lceil \text{load} / \text{target} \rceil$ using integer arithmetic.
2. **Bounds.** Clamp the recommendation to `[min_replicas, max_replicas]`. Call this clamped value `raw`.
3. **History.** Record `(now, raw)` in the history. At any later evaluation time `t`, a recorded entry from time `s` is *in the window* if `t - s < window_s`. The entry just recorded is always in the window.
4. **Scale up immediately.** If `raw >= current`, return `n = raw`.
5. **Scale down slowly.** Otherwise, return `n = min(current, highest recommendation in the window)`.

Under these rules, the replica count rises as soon as a single evaluation requests more capacity, and falls only to the highest recommendation made within the last `window_s` seconds.

Step 1 is the Kubernetes proportional rule $\lceil \text{current} \cdot (\text{metric} / \text{target}) \rceil$ applied to the metric "in-flight requests per replica, averaged over all starting and ready replicas." Because $\text{metric} = \text{load} / \text{current}$, the `current` factors cancel and give $\lceil \text{load} / \text{target} \rceil$. A replica that is still starting carries zero requests yet still counts in `current`.

**State.** Keep all persistent data inside `state` rather than global or module variables. Do not mutate the `state` object passed in, because the tests replay earlier states and expect identical results. The tests never inspect the internal structure of `state`.

**In the simulation.** Your policy drives `SimCluster`. Each replica runs at most 8 requests at once in first-come, first-served order, so `target = 6` leaves headroom. One simulation step costs $2^{-6}$ s plus $2^{-12}$ s per prefilled token plus $2^{-9}$ s per decoded sequence. Newly added replicas become ready 4 seconds later. Incoming requests go to the ready replica with the fewest requests, and removed replicas finish their in-flight requests before exiting. The simulation starts with 1 replica at $t = 0$, evaluates the policy every second, and uses `cfg = (target=6, tolerance=0.25, min_replicas=1, max_replicas=40, window_s=20)`.

## Example

Consider `target = 10`, `tolerance = 0.25`, `min_replicas = 1`, `max_replicas = 8`, and `window_s = 30`, evaluated every 10 seconds.

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

At `now = 20`, the four new replicas are still starting and the backlog remains high. Dividing `load = 52` by only the 2 ready replicas would give 26 per replica and trigger another scale-up. Dividing by all 6 replicas gives 8.7 per replica, which sits inside the tolerance band around 10. At `now = 30`, the entry from `now = 0` is exactly 30 seconds old (`30 - 0 == window_s`) and drops out of the window. The first scale-down occurs at `now = 50`, once the last recommendation of 6 (from `now = 20`) leaves the window. Even then, `n` drops to 5 rather than to the 2 recommended at `now = 40`.

## Constraints

- `1 <= target <= 1000` and `0 <= tolerance < 1`.
- `1 <= min_replicas <= max_replicas <= 1000`, `window_s >= 0`, and `ready + starting >= 1`.
- Evaluation times increase strictly, though they need not be evenly spaced.
- In the tests, `tolerance` is a multiple of 0.125 and times are multiples of 0.5, so every comparison is exact.
- `n` must be a Python `int`. The rules uniquely determine `n`, and tests check exact equality on both scripted inputs and simulations.
- **Load step workload:** 5 requests/s for 20 s, then 30 requests/s for 40 s, then 5 requests/s for 40 s (1478 requests, prompts of 8 to 64 tokens, outputs of 4 to 32 tokens), simulated for 120 s.
- **Bursts workload:** six bursts of 30 requests/s lasting 6 s each, spaced 6 s apart, on top of a 5 requests/s baseline (1463 requests), simulated for 132 s.
- Simulation limits across both workloads:

  | quantity | limit | reference, step | reference, bursts |
  |---|---|---|---|
  | requests whose first token took more than 1 s (or never came) | 160 step, 250 bursts | 111 | 127 |
  | replica-seconds paid for | 800 step, 900 bursts | 575 | 669 |
  | evaluations that changed the replica count | 18 | 13 | 13 |
  | largest `n` returned | 22 | 15 | 16 |

  The reference solution's late requests occur during the initial surge, before the first new replica finishes starting. No request may fail.

## Why this matters

GPU replicas dominate LLM serving costs and take tens of seconds to minutes to boot. Because of that startup delay, an autoscaler always acts on a cluster that has not yet absorbed its previous decision.

Three common policy bugs break production autoscalers. Ignoring replicas that are already starting causes the policy to re-order the same capacity on every tick and wildly overshoot. Scaling down as soon as traffic dips forces a cold start on every burst and causes constant SLO violations. Watching the wrong signal (such as GPU utilization or `running` requests, both of which saturate as soon as requests begin queueing) hides overload completely. The policy in this problem mirrors the Kubernetes Horizontal Pod Autoscaler driven by the metric LLM engines actually export for scaling: waiting requests plus running requests.
