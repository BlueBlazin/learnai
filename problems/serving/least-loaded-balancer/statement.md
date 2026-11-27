Implement power-of-two-choices load balancing for inference replicas, where each request picks two distinct replicas at random and routes to the less loaded one. Then build a stateful `Balancer` that keeps this policy effective even when replica load reports arrive periodically and become stale between updates.

![This problem is the Router, choosing the less loaded of two random replicas.](assets/map.svg)

## Specification

```python
def pick(replicas: list[int], load: dict[int, int], u1: float, u2: float) -> int: ...


class Balancer:
    def __init__(self, replicas: list[int]): ...
    def report(self, loads: dict[int, int]) -> None: ...
    def route(self, tokens: int, u1: float, u2: float) -> int: ...
```

**Load** is measured in outstanding tokens rather than request count. A replica's load is the total number of prompt tokens it still has to prefill plus output tokens it still has to generate across all requests it currently holds. Because LLM request lengths vary by orders of magnitude, counting requests alone does not reflect true work.

### `pick(replicas, load, u1, u2)`

`replicas` is a list of $n \ge 1$ distinct integer replica IDs in caller-defined order (not necessarily sorted or starting at 0). `load` maps each replica ID to its current outstanding tokens. `u1` and `u2` are independent uniform random numbers in $[0, 1)$ provided by the caller. `pick` must not use any other source of randomness.

1. If $n = 1$, return `replicas[0]`.
2. **First choice.** Pick index $i = \lfloor u_1 \cdot n \rfloor$, which is uniform over all $n$ replicas.
3. **Second choice (distinct from the first).** Pick $j = \lfloor u_2 \cdot (n - 1) \rfloor$. If $j \ge i$, increment $j \leftarrow j + 1$. This gives a uniform choice over the remaining $n - 1$ replicas.
4. **Compare.** Return `replicas[j]` if its load is **strictly** less than the load of `replicas[i]`. Otherwise, return `replicas[i]` so that ties go to the first choice.

In Python, `int(u * n)` computes the floor and is always strictly less than `n` when `0 <= u < 1`.

### `Balancer`

A balancer does not continuously observe each replica's true load. Instead, it receives periodic load reports and maintains a local load estimate for each replica between reports.

- `Balancer(replicas)`: store the replica IDs in the exact order given (this order determines how `pick` indexes them) and initialize every replica's estimated load to `0`.
- `report(loads)`: apply a new load report. `loads[replica_id]` is the true number of outstanding tokens on each replica at this instant, including any unfinished requests previously routed by this balancer. **Replace** every replica's estimate with its reported value.
- `route(tokens, u1, u2)`: choose a replica for a new request that adds `tokens` outstanding tokens (prompt plus output tokens). Call `pick(replicas, estimate, u1, u2)`, **add `tokens` to the chosen replica's estimate**, and return the chosen replica ID.

**Simulation setup.** Integration tests place your `Balancer` in front of the kit's `SimCluster` with 8 continuous-batching replicas. Each replica runs at most 8 sequences at once in first-come, first-served order, with each step taking $2^{-6}$ s plus $2^{-12}$ s per prefilled token plus $2^{-9}$ s per decoded sequence. A load report arrives at $t = 0$ and every `interval` seconds thereafter. If a report and a request arrive at the same instant, the report is delivered first. Requests arriving at the same instant are routed sequentially in arrival order, so each sees the estimates updated by the previous one.

## Example

```python
replicas = [0, 1, 2, 3, 4]
load = {0: 300, 1: 120, 2: 120, 3: 500, 4: 200}

pick(replicas, load, 0.1, 0.9)  # 4   first 0 (300), second 4 (200)
pick(replicas, load, 0.7, 0.1)  # 0   first 3 (500), second 0 (300)
pick(replicas, load, 0.5, 0.5)  # 2   first 2; j = 2 is shifted to 3 (500): 2 keeps it
pick(replicas, load, 0.3, 0.3)  # 1   first 1; j = 1 is shifted to 2: a tie, the first wins
pick(replicas, load, 0.5, 0.3)  # 2   first 2, second 1: the same tie, the first wins
```

Across all 20 ordered pairs of distinct replicas, replica 3 never wins, replica 0 wins only against replica 3 (2 pairs), and replica 4 wins against replicas 0 and 3 (4 pairs). Replicas 1 and 2 each win 6 pairs outright plus the 1 tied pair where they are drawn first. Their exact selection probabilities are $(0.1, 0.35, 0.35, 0, 0.2)$.

```python
b = Balancer([0, 1, 2])
b.route(100, 0.0, 0.0)  # 0   first 0, second 1: both estimates 0, a tie
b.route(50, 0.0, 0.0)  # 1   estimates 100 against 0
b.route(30, 0.0, 0.0)  # 1   100 against 50; replica 1 is now at 80
b.route(30, 0.4, 0.9)  # 2   first 1 (80), second 2 (0)
b.report({0: 10, 1: 80, 2: 30})  # replica 0 has nearly finished
b.route(40, 0.0, 0.9)  # 0   first 0 (10), second 2 (30); replica 0 is now at 50
b.route(40, 0.0, 0.9)  # 2   first 0 (50), second 2 (30)
```

## Constraints

- `1 <= n <= 1024`. Loads and `tokens` are integers in $[0, 10^9]$, and `u1`, `u2` are floats in $[0, 1)$.
- `pick` and `route` return a Python `int` from `replicas`. `pick` must not mutate its arguments.
- Routing decisions are checked for exact equality because `u1` and `u2` make every decision deterministic.
- Statistical tests draw 20,000 independent `(u1, u2)` pairs and compare empirical selection frequencies against the exact distribution (false-failure probability below $10^{-9}$).
- **Peak load.** On a seeded workload of 800 requests at 40 requests per second (prompts of 16 to 256 tokens, outputs of 8 to 64 tokens) with a report every 4 seconds, the maximum outstanding tokens held by any replica at any arrival must be at most **1100**. The reference solution peaks at 685. By comparison, uniform random routing peaks at 1847, and a balancer that ignores its own routing decisions between reports peaks at 2783.

## Why this matters

An inference cluster's tail latency is governed by its busiest replica, because requests routed to an overloaded GPU wait behind long generations while other GPUs sit idle. Always routing to the globally least-loaded replica sounds ideal, but it fails when load metrics are even slightly stale. Every router sees the same "idle" replica and floods it at once.

Uniform random routing avoids herding but leaves large load imbalances. Checking two random replicas achieves almost the full balance of a global scan with only two lookups, and it degrades gracefully under stale metrics. That is why Envoy uses power-of-two-choices as its default least-request policy and why LLM gateways build cache-aware routers on top of it. Measuring load in tokens rather than requests is equally important, since a replica holding one 30,000-token prompt is far busier than one holding a 20-token request.
