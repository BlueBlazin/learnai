## The idea

**Two choices.** Send each request to a uniformly random replica and the loads
spread like balls thrown into bins: the fullest bin holds far more than the average.
Look at two random bins and use the emptier one, and the imbalance collapses: the
gap between the fullest bin and the average grows only doubly logarithmically in the
number of bins, an exponential improvement over one choice ("The Power of Two Choices
in Randomized Load Balancing", IEEE Transactions on Parallel and Distributed Systems,
2001). A third choice improves on two only by a constant factor, and a full scan of
all replicas is barely better than two. The second lookup is where the gain is.

**Two *distinct* choices.** Drawing both indices independently returns the same
replica twice with probability $1/n$, and that request is routed at random. Drawing
the second from the other $n - 1$ avoids the waste. One uniform number is enough:
choose a position among the $n - 1$ remaining and step over the first choice,

```python
j = int(u2 * (n - 1))
if j >= i:
    j += 1
```

which maps $\{0, \dots, n-2\}$ one-to-one onto the indices other than $i$.

**Ties.** With token counts as load, exact ties look rare, and they are not: every
idle replica has load 0, and a balancer's own estimates all start at 0. If a tie goes
to the lower id, then among $n$ idle replicas the lowest id wins every pair it
appears in, probability $2/n$, and the highest id never wins: it receives no traffic
until the others have work. Giving the tie to the first draw is fair without any
extra randomness, because each of the two tied replicas is equally likely to have
been drawn first.

**Stale load.** A balancer learns the replicas' load from reports that arrive
periodically, or from a metrics scrape, and in between it knows nothing new. If it
routes on the last report alone, every request until the next report sees the same
picture, and the replicas that looked idle receive everything: load information of
age $T$ turns "least loaded" into "least loaded $T$ seconds ago, and most loaded
now". The repair is to add what the balancer itself knows: it sent these tokens to
that replica since the report. The estimate is then

$$
\text{estimate} = \text{last report} + \text{tokens routed since},
$$

and a new report replaces it wholesale, because the report already includes those
requests (or they have finished).

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

One seeded run on the kit's `SimCluster` (8 replicas, 800 requests at 40 per second,
a report every 4 seconds; a different seed from the test's):

| policy | peak outstanding tokens on one replica | p99 TTFT |
|---|---|---|
| least loaded of all, on the last report only | 26,432 | 24.2 s |
| two choices, on the last report only | 3,939 | 3.5 s |
| uniform random | 1,915 | 1.5 s |
| two choices, with its own routing added (this problem) | 1,062 | 0.63 s |
| least loaded of all, with its own routing added | 906 | 0.40 s |

The first row is the herd: for four seconds at a time every request goes to the
replica that was emptiest at the last report. Stale information used greedily is
much worse than no information. Two choices on the same stale reports is already
six times better, because the randomness spreads the herd over the replicas that
compare well; with local accounting it beats random routing by a wide margin. The
last row is a reminder of what this simulation does not contain: one balancer that
accounts for all of its own traffic can afford a full scan. With several balancer
instances, none of which sees the others' decisions, the greedy rule herds again
and two choices does not, which is why it is the default.

## Common mistakes

- **The same replica drawn twice.** `int(u2 * n)` for the second index. One request
  in $n$ gets no comparison, and the most loaded replica can be chosen (when it is
  drawn twice), which with distinct draws is impossible.
- **Forgetting the shift.** `int(u2 * (n - 1))` without `if j >= i: j += 1` can
  repeat the first choice and can never select the last replica.
- **Ties to the lower id.** Deterministic, reproducible, and biased: the highest id
  is starved whenever replicas are idle or equally loaded.
- **Ties to the second draw.** Equally fair in distribution, but not the specified
  rule: with the uniforms given the decision is determined, and `<=` flips it.
- **Routing at random.** Returning the first draw passes every test of the draws and
  ignores the load.
- **Sorting the replicas.** The caller's order defines which replica a uniform
  selects.
- **No local accounting.** Routing on the last report alone; see the table.
- **Counting requests.** Adding 1 per routed request instead of its tokens makes a
  4,000-token prompt weigh as much as a 20-token one. (It also mixes units with the
  reports, which are in tokens.)
- **Adding reports to the estimate.** A report is a new measurement, not an
  increment. Accumulating it counts every outstanding request twice.

## Complexity

`pick` and `route` are $O(1)$; `report` is $O(n)$.

## In the wild

- **Envoy**'s least-request load balancer uses power of two choices when all hosts
  have equal weight: `choice_count` random hosts (2 by default), the one with the
  fewest active requests wins. Its implementation draws each host independently, so
  the same host can be drawn twice, and a later draw replaces the candidate only if
  it has strictly fewer active requests, so ties go to the first drawn, as here
  (`source/extensions/load_balancing_policies/least_request/least_request_lb.cc`).
  Its full-scan mode breaks ties uniformly at random. Envoy counts requests; its
  counts are its own local accounting and are never stale within one proxy.
- **Stale information** was analyzed in "How Useful Is Old Information?" (IEEE
  Transactions on Parallel and Distributed Systems, 2000): with load reports of age
  $T$, sending to the apparently least loaded server can be worse than choosing at
  random, and choosing the better of two random servers remains effective.
- **LLM gateways** combine this with what is specific to inference: load in tokens or
  in queue depth scraped from the engine's metrics, and a preference for the replica
  that already holds the prompt's prefix in its KV cache, traded against load.
