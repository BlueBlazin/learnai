## The idea

A replica's prefix cache is only useful to requests that reach that replica. With
$n$ replicas behind a load-only balancer, a request finds its prefix cached with
probability roughly $1/n$ of what a single server would give, unless the cache is
large enough to hold every prefix everywhere. So the router should send requests
with the same beginning to the same place: *affinity*.

**What to match on.** The KV cache stores keys and values per position, and position
$i$ depends on all tokens before it. Two prompts can therefore share cached work
only up to the first token at which they differ. The useful quantity is the length
of the longest common prefix, measured in tokens, and its fraction of the prompt
(the *match rate*): a 30-token match on a 4000-token prompt saves nothing worth
giving up balance for, which is what the threshold expresses.

**An index of your own.** The router cannot ask the replicas what they hold on
every request. It knows what it sent where, and a prompt that was routed to a
replica will be in that replica's cache shortly. So it keeps, per replica, the
recent prompts it routed there and matches against those. This index is an
approximation in two ways, and both matter:

- It is early: the prompt is recorded when it is routed, before the replica has
  prefilled it. That is deliberate. Two requests with the same prefix arriving
  together should go to the same replica even though the first has not been cached
  yet; an index fed only by confirmations from the replicas would split them.
- It must forget. The replica evicts; an index that remembers everything believes
  in cache entries that are long gone and follows them. Bounding the record to
  about what the replica's cache can hold keeps the two in step.

**Bounded load.** Pure affinity turns popularity into overload: with skewed
traffic the hottest prefix is a large share of all requests, and they all go to
one replica. The repair is to let affinity choose only among replicas whose load is
within a bound of the least loaded one. When the preferred replica is over the
bound, the request goes elsewhere, and, because the index records it there, the
prefix now has a second home. Later requests choose between the two homes by load.
A hot prefix thus spreads over exactly as many replicas as its traffic needs, and a
cold one stays on one. This is the idea of consistent hashing with bounded loads
("Consistent Hashing with Bounded Loads", arXiv:1608.01350: no server above
$\lceil c \cdot m / n \rceil$ for a chosen $c > 1$, overflow goes to the next
server), applied to a longest-prefix match instead of a hash.

## Reference solution (Python)

```python
from collections import deque


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class Router:
    def __init__(self, match_threshold, imbalance, index_size):
        self.match_threshold = match_threshold
        self.imbalance = imbalance
        self.index_size = index_size
        self.index = {}

    def match(self, tokens, replica_id):
        prompts = self.index.get(replica_id, ())
        return max((common_prefix(tokens, prompt) for prompt in prompts), default=0)

    def route(self, tokens, replicas):
        least = min(replica.queue_len for replica in replicas)
        eligible = [r for r in replicas if r.queue_len - least <= self.imbalance]
        best = max(eligible, key=lambda r: (self.match(tokens, r.id), -r.queue_len, -r.id))
        if self.match(tokens, best.id) >= self.match_threshold * len(tokens):
            chosen = best
        else:
            chosen = min(replicas, key=lambda r: (r.queue_len, r.id))
        self.index.setdefault(chosen.id, deque(maxlen=self.index_size)).append(tuple(tokens))
        return chosen.id
```

## What the bound buys

One seeded run on the cluster of the statement (600 requests, 8 prefixes; a
different seed from the tests'). "Gap" is the largest difference, at a routing
decision, between the queue of the chosen replica and the shortest queue.

| workload | router | cache-hit tokens | p95 TTFT | gap |
|---|---|---|---|---|
| uniform, 30/s | load only | 56,592 | 1.41 s | 0 |
| | this problem | 151,792 | 0.28 s | 4 |
| | affinity, no bound | 165,808 | 0.30 s | 11 |
| | index that never forgets | 107,728 | 0.47 s | 4 |
| hot prefix, 40/s | load only | 113,088 | 0.67 s | 0 |
| | this problem | 154,704 | 0.27 s | 4 |
| | affinity, no bound | 180,288 | 9.15 s | 166 |
| | index that never forgets | 142,336 | 0.31 s | 4 |

With evenly popular prefixes, unbounded affinity is harmless: the prefixes
themselves balance the load. With one hot prefix it has the best hit rate of all
and a tail latency thirty times worse, because cache hits do not help a request
that waits behind 160 others. Note also the first row of each block: the load-only
router is not at zero. It hits whenever a replica happens to still hold the prefix,
which for a hot prefix is often.

## Common mistakes

- **Affinity never broken.** No eligibility test, or one that is never false. Passes
  every uniform benchmark and fails in production on the day one tenant's system
  prompt becomes popular.
- **The bound applied to the queue itself.** `queue_len <= imbalance` instead of
  `queue_len - least <= imbalance`: under load nobody is eligible and the router
  silently degrades to load-only. The bound is about imbalance, not about load.
- **The index not updated**, or updated only when the match was used. Then the
  first request of a prefix leaves no trace and the second cannot follow it.
- **Matching tokens in common rather than a prefix**: counting equal positions
  without stopping at the first difference, or requiring identical prompts. The
  first over-counts (templates repeat), the second never matches chat traffic,
  where the last message always differs.
- **Ties among equal matches by position.** Once a prefix lives on two replicas, the
  choice between them is the load-balancing decision for that prefix.
- **An index that never forgets**; see the table.
- **Falling back to something other than the least loaded replica** when the match
  is low.

## Complexity

`route` compares the prompt with up to `index_size` prompts per replica:
$O(R \cdot S \cdot L)$ token comparisons for $R$ replicas, index size $S$ and prompt
length $L$. Production routers replace the scan with a prefix tree whose nodes
record which replicas hold them, or with hashes of fixed-size token blocks chained
from the start of the prompt (the same block hashes the engine's prefix cache
uses), which makes the lookup $O(L)$ regardless of $R$ and $S$.

## In the wild

- **Ray Serve LLM**'s `PrefixCacheAffinityRouter` keeps a prefix tree of the prompts
  routed to each replica (on the prompt's characters, not tokens). If the difference
  between the longest and shortest replica queue exceeds `imbalanced_threshold`, it
  routes by power of two choices; otherwise it finds the best prefix match and uses
  it when the match rate reaches `match_rate_threshold` (default 0.1), and sends
  low-match requests to the replica with the lowest cache usage. Its imbalance test
  is global (the whole cluster is balanced or not), where this problem's is per
  candidate (Ray documentation, "Prefix-aware routing").
- The **SGLang router**'s cache-aware policy keeps an approximate radix tree per
  worker, mirroring the radix tree of the worker's prefix cache, with periodic LRU
  eviction. It is in load-balancing mode when the load difference exceeds both an
  absolute and a relative threshold (`balance_abs_threshold`,
  `balance_rel_threshold`), and otherwise routes to the best prefix match if the
  match ratio exceeds `cache_threshold`, else to the worker with the most free
  cache (SGLang documentation, "SGLang Router").
- **llm-d** (on the Kubernetes Gateway API Inference Extension) scores pods by
  prefix-cache match and combines that score with queue-depth and KV-utilization
  scorers. Its approximate mode tracks routing decisions, like this problem; its
  precise mode builds the index from cache events published by the model servers,
  and adds "speculative" entries at routing time for exactly the reason given
  above: confirmed events arrive after back-to-back requests have already been
  routed (llm-d documentation, "Prefix-Cache Aware Routing" and "Precise Prefix
  Cache Routing").
