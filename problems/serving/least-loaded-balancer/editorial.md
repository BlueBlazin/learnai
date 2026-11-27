## The idea

**Power of two choices.** Under uniform random routing, load spreads like balls thrown independently into bins, leaving the busiest bin far above the average. Sampling two random bins and choosing the emptier one collapses this imbalance so that the gap between the fullest bin and the average grows only doubly logarithmically in the number of bins ("The Power of Two Choices in Randomized Load Balancing", IEEE Transactions on Parallel and Distributed Systems, 2001). Adding a third choice improves only by a constant factor, and scanning all replicas is barely better than checking two.

**Two distinct choices.** If both indices are drawn independently from all $n$ replicas, the same replica is picked twice with probability $1/n$, wasting the comparison. Drawing the second index from the remaining $n - 1$ replicas avoids duplicates without a retry loop. Pick an index in $\{0, \dots, n - 2\}$ and shift it past the first choice `i`:

```python
j = int(u2 * (n - 1))
if j >= i:
    j += 1
```

This maps $\{0, \dots, n - 2\}$ bijectively onto the $n - 1$ indices other than `i`.

**Breaking ties fairly.** Exact ties happen constantly in practice because idle replicas all have load `0`, and a new `Balancer` initializes every estimate to `0`. Breaking ties in favor of the lower replica ID starves higher IDs. Among $n$ idle replicas, the lowest ID wins every pair it appears in (probability $2/n$), while the highest ID never wins. Breaking ties in favor of the first random draw is fair with no extra randomness because either tied replica is equally likely to be drawn first.

**Handling stale reports.** Between periodic load reports or metrics scrapes, the balancer receives no feedback from replicas. If it routes solely on the last report, every request until the next report sees the same stale snapshot and piles onto whichever replicas looked empty at the last report. Adding locally routed tokens to each replica's estimate fixes this:

$$
\text{estimate} = \text{last report} + \text{tokens routed since}
$$

When a fresh report arrives, it replaces the estimate completely because the replica's true count already reflects all unfinished requests.

## Reference solution (Python)

```python
def pick(replicas, load, u1, u2):
    n = len(replicas)
    if n == 1:
        return replicas[0]
    i = int(u1 * n)
    j = int(u2 * (n - 1))
    if j >= i:
        j += 1
    first, second = replicas[i], replicas[j]
    return second if load[second] < load[first] else first


class Balancer:
    def __init__(self, replicas):
        self.replicas = list(replicas)
        self.estimate = dict.fromkeys(self.replicas, 0)

    def report(self, loads):
        self.estimate = {replica: loads[replica] for replica in self.replicas}

    def route(self, tokens, u1, u2):
        chosen = pick(self.replicas, self.estimate, u1, u2)
        self.estimate[chosen] += tokens
        return chosen
```

## What staleness does

Here is a seeded simulation run on the kit's `SimCluster` with 8 replicas, 800 requests at 40 requests per second, and a report every 4 seconds (using a different seed from the test suite):

| Policy | Peak outstanding tokens on one replica | p99 TTFT |
|---|---|---|
| Least loaded of all, on the last report only | 26,432 | 24.2 s |
| Two choices, on the last report only | 3,939 | 3.5 s |
| Uniform random | 1,915 | 1.5 s |
| Two choices, with local accounting (this problem) | 1,062 | 0.63 s |
| Least loaded of all, with local accounting | 906 | 0.40 s |

The first row shows the herd effect. For four seconds at a time, every arriving request goes to the single replica that looked emptiest at the last report, making greedy routing on stale data far worse than random routing. Power-of-two-choices on the same stale reports is already six times better because random sampling spreads traffic across several lightly loaded replicas. Adding local accounting beats uniform random routing by a wide margin.

In the last row, a single centralized balancer that tracks all of its own traffic can afford a full scan. However, production systems run multiple independent balancer instances that cannot see each other's in-flight decisions. In that setting, greedy routing herds again while power-of-two-choices stays balanced.

## Common mistakes

- **Drawing the same replica twice.** Computing `int(u2 * n)` for the second index skips the comparison on $1/n$ of requests and allows the single busiest replica to be chosen.
- **Forgetting the index shift.** Using `int(u2 * (n - 1))` without `if j >= i: j += 1` can still repeat `i` and can never pick the last replica.
- **Breaking ties by lower ID.** This is deterministic but biased, starving the highest replica ID whenever replicas are idle or equally loaded.
- **Breaking ties in favor of the second draw.** Using `<=` instead of `<` is statistically fair, but it violates the specification and fails exact deterministic checks.
- **Ignoring load and returning the first draw.** This passes distribution checks on uniform draws while failing load-balancing tests.
- **Sorting `replicas`.** The caller's list order defines which replica each uniform value selects, so reordering changes the output.
- **Omitting local accounting.** Routing solely on the last report causes herding between updates, as shown in the table above.
- **Counting requests instead of tokens.** Adding `1` per routed request treats a 4,000-token prompt the same as a 20-token prompt and mixes units with token-based reports.
- **Accumulating reports onto existing estimates.** Each report is a full snapshot rather than a delta, so adding it double-counts every outstanding request.

## Complexity

`pick` and `route` run in $O(1)$ time, and `report` runs in $O(n)$ time to refresh all $n$ replica estimates.

## In the wild

- **Envoy.** Envoy's least-request load balancer uses power-of-two-choices when hosts have equal weight (`choice_count` defaults to 2). Its implementation (`source/extensions/load_balancing_policies/least_request/least_request_lb.cc`) draws hosts independently (so duplicates can occur) and replaces the candidate only when a later draw has strictly fewer active requests, breaking ties in favor of the first draw just like this problem. In full-scan mode, it breaks ties uniformly at random. Because Envoy counts active requests locally within each proxy, its counts are never stale for that proxy's own traffic.
- **Stale metrics.** Mitzenmacher analyzed delayed load reports in "How Useful Is Old Information?" (IEEE Transactions on Parallel and Distributed Systems, 2000), showing that greedy routing on metrics of age $T$ can perform worse than uniform random routing, whereas power-of-two-choices remains effective.
- **LLM gateways.** Inference routers extend power-of-two-choices with LLM-specific signals, combining token load or queue depth scraped from engine metrics with KV-cache prefix affinity so requests prefer replicas that already cache their prompt prefix without overloading them.
