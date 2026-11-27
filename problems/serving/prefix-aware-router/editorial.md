## The idea

A replica's prefix cache only helps requests that actually land on that replica. Behind a load-only balancer with $n$ replicas, a request finds its prefix cached roughly $1/n$ as often as on a single server, unless every replica's cache is large enough to hold all active prefixes. Sending requests with the same beginning to the same replica (*prefix affinity*) restores those cache hits.

**What to match on.** A KV cache stores keys and values at each token position, and position $i$ attends to all tokens before it. Two prompts can therefore reuse cached KV entries only up to the first position where they differ. The right metric is the length of the longest common prefix in tokens, expressed as a fraction of the prompt length (the *match rate*). A 30-token match on a 4000-token prompt saves too little prefill work to justify hurting load balance, which is why `match_threshold` filters out weak matches.

**Keeping a local index.** The router cannot query every replica's cache on every request. Instead, it tracks what it recently sent to each replica, because a routed prompt will soon be in that replica's cache. This local index is an approximation in two deliberate ways:

- **It updates early.** The prompt is recorded at routing time, before the replica finishes prefilling it. When two requests with the same prefix arrive back to back, both route to the same replica. An index that waited for replica confirmations would split them.
- **It forgets old entries.** Replicas evict old prefixes from their caches. An unbounded index would keep chasing cache entries that were evicted long ago. Capping each replica's record to `index_size` (roughly what its cache holds) keeps the router's view aligned with the replica.

**Bounded load.** Pure affinity turns traffic skew into overload. If one prefix dominates traffic, every request using it piles onto a single replica. Restricting affinity to replicas within `imbalance` requests of the least loaded replica fixes this. Once the preferred replica exceeds the bound, the request overflows to a less busy replica. Because the index records that overflow, the prefix now has two homes, and subsequent requests pick whichever home has the shorter queue. Hot prefixes spread across just enough replicas to absorb their load, while cold prefixes stay on one replica. This applies the idea of consistent hashing with bounded loads ("Consistent Hashing with Bounded Loads", arXiv:1608.01350, where distributing $m$ requests across $n$ servers caps each server at $\lceil c \cdot m / n \rceil$ for a chosen $c > 1$ and overflows to the next server) to longest-prefix matching instead of hashing.

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

The table below shows one seeded run on the cluster from the statement (600 requests, 8 prefixes, using a different seed from the test suite). "Gap" is the largest difference, at any routing decision, between the chosen replica's queue length and the shortest queue length.

| workload | router | cache-hit tokens | p95 time to first token | gap |
|---|---|---|---|---|
| uniform, 30/s | load only | 56,592 | 1.41 s | 0 |
| | this problem | 151,792 | 0.28 s | 4 |
| | affinity, no bound | 165,808 | 0.30 s | 11 |
| | index that never forgets | 107,728 | 0.47 s | 4 |
| hot prefix, 40/s | load only | 113,088 | 0.67 s | 0 |
| | this problem | 154,704 | 0.27 s | 4 |
| | affinity, no bound | 180,288 | 9.15 s | 166 |
| | index that never forgets | 142,336 | 0.31 s | 4 |

When prefixes are equally popular, unbounded affinity looks harmless because the prefixes naturally balance the load. With one hot prefix, unbounded affinity gets the highest cache-hit count but 30 times worse tail latency, because a cache hit cannot save a request stuck behind 160 queued requests. Notice also that the load-only router is not at zero hits. It still gets hits whenever a replica happens to retain a prefix, which happens often for a hot prefix.

## Common mistakes

- **Never breaking affinity.** Omitting the eligibility check, or writing one that is always true, passes uniform benchmarks and fails in production as soon as one tenant's system prompt becomes popular.
- **Comparing absolute queue length to `imbalance`.** Checking `queue_len <= imbalance` instead of `queue_len - least <= imbalance` makes every replica ineligible under high load, silently degrading the router to load-only. The bound constrains queue imbalance, not absolute queue depth.
- **Not updating the index on every route.** Updating the index only when a prefix match wins leaves the first request of a new prefix unrecorded, so the second request cannot follow it.
- **Counting matching tokens instead of a common prefix.** Counting equal positions without stopping at the first mismatch over-counts shared template tokens. Conversely, requiring exact prompt equality fails on chat workloads where the latest turn always differs.
- **Breaking equal-match ties by list position.** Once a hot prefix spreads to two replicas, breaking ties by `queue_len` is what balances load between them.
- **Never evicting from the index.** Keeping every historical prompt degrades hit rate and latency, as shown in the table above.
- **Wrong fallback on low match.** When the best match is below `match_threshold * len(tokens)`, the router must fall back to the least loaded replica across all of `replicas`, not an arbitrary candidate.

## Complexity

`route` compares the incoming prompt against up to `index_size` prompts for each replica. For $R$ replicas, index size $S$, and prompt length $L$, this takes $O(R \cdot S \cdot L)$ token comparisons. Production routers replace this linear scan with a prefix tree (whose nodes record which replicas hold each prefix) or with chained hashes of fixed-size token blocks matching the engine's prefix cache, reducing lookup time to $O(L)$ independent of $R$ and $S$.

## In the wild

- **Ray Serve LLM** (`PrefixCacheAffinityRouter`) maintains a character-level prefix tree of prompts routed to each replica. When the gap between the longest and shortest replica queues exceeds `imbalanced_threshold`, it switches to power-of-two-choices load balancing. Otherwise it picks the best prefix match if the match rate reaches `match_rate_threshold` (default 0.1), and sends low-match requests to the replica with the lowest cache usage. Unlike this problem's per-candidate eligibility check, Ray's imbalance check is global across the cluster (Ray documentation, "Prefix-aware routing").
- **SGLang Router** keeps an approximate radix tree per worker with periodic LRU eviction, mirroring each worker's prefix cache. It switches to load-balancing mode when the load difference exceeds both an absolute threshold (`balance_abs_threshold`) and a relative threshold (`balance_rel_threshold`). Otherwise it routes to the best prefix match if the match ratio exceeds `cache_threshold`, or falls back to the worker with the most free cache (SGLang documentation, "SGLang Router").
- **llm-d** (built on the Kubernetes Gateway API Inference Extension) scores pods by prefix-cache match and combines that score with queue-depth and KV-utilization scorers. Its approximate mode tracks routing decisions just like this problem. Its precise mode builds the index from cache events published by the model servers, and inserts speculative entries at routing time so back-to-back requests with the same prefix land on the same replica before the confirmation event arrives (llm-d documentation, "Prefix-Cache Aware Routing" and "Precise Prefix Cache Routing").
