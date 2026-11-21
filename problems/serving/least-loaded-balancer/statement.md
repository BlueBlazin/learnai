Implement the routing rule that most production load balancers use in front of
replicas: draw two replicas at random and send the request to the less loaded one.
Then make it work when the load information is old. The rule is three lines; the
parts that decide whether it balances are how the two draws are made, who wins a tie,
and what the balancer believes between load reports.

![This problem is the Router, choosing the less loaded of two random replicas.](assets/map.svg)

## Specification

```python
def pick(replicas: list[int], load: dict[int, int], u1: float, u2: float) -> int: ...


class Balancer:
    def __init__(self, replicas: list[int]): ...
    def report(self, loads: dict[int, int]) -> None: ...
    def route(self, tokens: int, u1: float, u2: float) -> int: ...
```

**Load** is counted in outstanding tokens: for one replica, the prompt tokens it
still has to prefill plus the output tokens it still has to produce, over all the
requests it holds. Requests differ in size by orders of magnitude, so the number of
requests is not the load.

### `pick(replicas, load, u1, u2)`

`replicas` is a list of $n \ge 1$ distinct replica ids, used in the order given (it
need not be sorted or start at 0). `load` maps each id to its outstanding tokens.
`u1` and `u2` are independent uniform numbers in $[0, 1)$ supplied by the caller;
`pick` uses no other randomness.

1. If $n = 1$, return the only replica.
2. **First choice**: index $i = \lfloor u_1 \cdot n \rfloor$, uniform over all $n$.
3. **Second choice**, different from the first: $j = \lfloor u_2 \cdot (n - 1)
   \rfloor$, and if $j \ge i$ then $j \leftarrow j + 1$. This is uniform over the
   other $n - 1$ replicas.
4. Return `replicas[j]` if its load is **strictly** smaller than that of
   `replicas[i]`; otherwise return `replicas[i]`. A tie goes to the first choice.

In Python `int(u * n)` is the floor and is always below `n` for `u < 1`.

### `Balancer`

A balancer does not see the replicas' true load. It receives a **report** from time
to time and keeps an **estimate** per replica in between.

- `Balancer(replicas)`: the replica ids, in the order `pick` will index them. Every
  estimate starts at 0.
- `report(loads)`: a load report has arrived. `loads[id]` is the true number of
  outstanding tokens of each replica at this instant, including whatever this
  balancer sent earlier that is still unfinished. Every estimate is **replaced** by
  the reported value.
- `route(tokens, u1, u2)`: route one request that will add `tokens` outstanding
  tokens to the replica it goes to (its prompt tokens plus its output tokens). The
  balancer returns `pick(replicas, estimate, u1, u2)` and **adds `tokens` to the
  estimate of the chosen replica**, so that its next decision knows about this one.

**In the simulation**, the tests put a `Balancer` in front of the kit's `SimCluster`:
8 replicas, each a continuous-batching server that runs at most 8 sequences at once
(first come, first served; a step costs $2^{-6}$ s plus $2^{-12}$ s per prefilled
token plus $2^{-9}$ s per decoded sequence). A report is delivered at $t = 0$ and
then every `interval` seconds; a report due at the same instant as an arriving
request is delivered before the request is routed. Requests arriving at the same
instant are routed one after another in arrival order, each seeing the estimates left
by the previous one.

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

Over all 20 ordered pairs of distinct replicas, replica 3 never wins, replica 0 wins
only against 3 (2 pairs), replica 4 against 0 and 3 (4 pairs), and replicas 1 and 2
each win 6 pairs outright plus the one tied pair in which they were drawn first. The
choice probabilities are therefore exactly $(0.1, 0.35, 0.35, 0, 0.2)$.

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

- `1 <= n <= 1024`; loads and `tokens` are ints in $[0, 10^9]$; `u1`, `u2` in
  $[0, 1)$.
- `pick` and `route` return a Python int that is one of the ids. `pick` does not
  modify its arguments.
- Routing decisions are compared exactly: with the uniforms given, every decision is
  determined.
- Statistical cases draw 20,000 independent pairs `(u1, u2)` and compare the choice
  frequencies with the exact distribution (false-failure probability below
  $10^{-9}$).
- **Peak load.** On a seeded workload of 800 requests at 40 requests per second
  (prompts of 16 to 256 tokens, outputs of 8 to 64) with a report every 4 seconds,
  the largest number of outstanding tokens any replica holds, observed at each
  arrival, must be at most **1100**. The reference peaks at 685. Uniform random
  routing peaks at 1847, and a balancer that does not account for its own decisions
  between reports at 2783.

## Why this matters

A fleet of inference replicas is only as fast as its most loaded member: a request
routed to a busy replica waits behind long generations while other GPUs idle.
Sending each request to the least loaded replica sounds optimal and is fragile,
because the load a router sees is always a little old, and every router that sees
the same idle replica sends to it at once. Random routing has no such herd and no
balance either. Two random choices get nearly all the balance of a full scan at the
cost of two lookups, and degrade gracefully when the information is stale. That is
how Envoy's least-request policy selects a host by default, and it is the starting
point of the cache-aware routers in LLM gateways. For LLM traffic the unit matters as much as the
policy: a replica holding one 30,000-token request is not as free as one holding a
20-token request, and a balancer that counts requests cannot tell them apart.
