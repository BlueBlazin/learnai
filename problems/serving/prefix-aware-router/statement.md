Implement a `Router` that sends each request to the replica most likely to hold the start of its prompt in its KV cache. Reusing that cached prefix avoids prefilling shared system prompts again, while a load bound prevents a popular prefix from piling every request onto one replica. Because the router cannot inspect the replicas' caches directly, it tracks its own recent routing history and balances prefix match length against queue depth.

![This problem is the Router, sending a request to the replica that likely has its prefix cached.](assets/map.svg)

## Specification

```python
class Router:
    def __init__(self, match_threshold: float, imbalance: int, index_size: int): ...
    def route(self, tokens: tuple[int, ...], replicas: list) -> int: ...
```

The constructor configures three parameters:

- `match_threshold`: minimum fraction of the prompt that must match for prefix affinity to apply, in `[0, 1]`.
- `imbalance`: maximum queue-length gap above the least loaded replica that an eligible replica may have.
- `index_size`: maximum number of recent prompts the router remembers per replica.

`route` is called once per request, in arrival order. `tokens` is a non-empty tuple of prompt token IDs. `replicas` is a non-empty list of candidate replicas in arbitrary order. Each replica exposes an integer `id` (unique) and an integer `queue_len` (the number of requests it currently holds, waiting or running). `route` returns the chosen replica's `id` as a Python `int`.

**The index.** For each replica `id`, the router stores the last `index_size` prompts routed to it, dropping the oldest entry when the record exceeds `index_size`. Routing a prompt to a replica always appends it, even if an identical prompt is already stored (so duplicates count toward `index_size`). The index starts empty, and any previously unseen replica `id` has an empty record.

**The match.** The match score between `tokens` and a replica is the length of the longest common prefix between `tokens` and any prompt stored for that replica. Two prompts' common prefix is the number of leading positions where they agree, stopping at the first mismatch or at the end of either prompt. An empty record has a match score of `0`.

**The decision.** Let `least` be the smallest `queue_len` among the replicas in `replicas`.

1. A replica is **eligible** if `queue_len - least <= imbalance`.
2. Among all eligible replicas, pick the **best** replica by the largest match score. Break ties by the smaller `queue_len`, then by the smaller `id`.
3. If `best`'s match score is at least `match_threshold * len(tokens)`, choose `best`.
4. Otherwise, fall back to the least loaded replica across all of `replicas`, breaking ties by the smaller `id`.
5. Append `tokens` to the chosen replica's record and return its `id`.

Consider only the replicas passed to the current `route` call. Keep stored records for any other replica IDs in case those replicas appear in later calls.

**In the simulation.** The router sits in front of `SimCluster` with four replicas. Each replica is a continuous-batching server that runs at most 8 sequences at once in first-come, first-served order. One step takes $2^{-6}$ s plus $2^{-12}$ s per prefilled token plus $2^{-9}$ s per decoded sequence. Each replica maintains a 1200-token prefix cache (about three prompts) with LRU eviction. When a request is admitted, the longest prefix it shares with any prompt that replica has already prefilled (rounded down to whole 16-token blocks) skips prefill. The simulation constructs `Router(0.25, 4, 3)`. Your router must not read any attribute from a replica other than `id` and `queue_len`.

## Example

Consider `Router(match_threshold=0.5, imbalance=2, index_size=2)` with three replicas. Queue lengths are shown as `{id: queue_len}`.

| call | `tokens` | queues | returns | why |
|---|---|---|---|---|
| 1 | `(1, 2, 3, 4, 5, 6, 7, 8)` | `{0: 0, 1: 0, 2: 0}` | 0 | No prompts are remembered yet. Pick the least loaded replica with the lowest `id`. |
| 2 | `(1, 2, 3, 4, 5, 6, 9, 9)` | `{0: 1, 1: 0, 2: 0}` | 0 | Replica 0 matches 6 of 8 tokens ($6 \ge 0.5 \cdot 8$) and is only 1 request above `least`. |
| 3 | `(1, 2, 3, 7, 7, 7, 7, 7)` | `{0: 2, 1: 0, 2: 0}` | 1 | The best match is 3 tokens, which is below 4. Fall back to the least loaded replica with the lowest `id`. |
| 4 | `(1, 2, 3, 4, 5, 6, 7, 7)` | `{0: 3, 1: 1, 2: 0}` | 2 | Replica 0 matches 7 tokens but is 3 above `least`, so it is ineligible. Replica 1 matches only 3 tokens. Fall back to least loaded (replica 2). |
| 5 | `(1, 2, 3, 4, 5, 6, 7, 0)` | `{0: 2, 1: 1, 2: 1}` | 2 | Replicas 0 and 2 both match 7 tokens and are eligible. Replica 2 wins on shorter `queue_len`. |

After call 4, both replica 0 and replica 2 remember the prefix `(1, 2, 3, 4, 5, 6, 7)`. Call 5 then picks whichever of the two is less busy. In this way, a prefix that is too hot for one replica spills onto a second replica without spreading further than needed.

## Constraints

- `0 <= match_threshold <= 1`, `imbalance >= 0`, and `index_size >= 1`.
- Up to 16 replicas and prompts of up to 4096 tokens.
- In direct tests, `match_threshold` is a multiple of 0.25, so `match_threshold * len(tokens)` is exact.
- The rules above uniquely determine every routing decision. Decisions are checked for exact equality on scripted calls and on seeded simulation workloads of 300 to 600 requests, where each prompt combines a shared prefix (64 to 512 tokens) with a unique suffix.
- **Against a load-only router** (smallest `queue_len`, lowest `id` on ties) on the same cluster with 600 requests across 8 prefixes of 256 to 512 tokens:

  | workload | cache-hit tokens | p95 time to first token |
  |---|---|---|
  | prefixes equally likely, 30 requests/s | at least 1.8 times the baseline's | at most 0.6 times the baseline's |
  | one hot prefix (weights $(g+1)^{-2}$ for prefix $g = 0, \dots, 7$), 40 requests/s | at least 1.1 times | at most 0.75 times |

  The reference solution achieves 2.9 times and 0.13 times on the uniform workload, and 1.29 times and 0.45 times on the hot-prefix workload (where the load-only baseline already gets frequent cache hits because the hot prefix lands on every replica).
- On both workloads, the chosen replica is never more than `imbalance` = 4 requests above the least loaded replica at that moment. By contrast, a router that always follows the best match sends requests to a replica 156 requests above the least loaded one on the hot-prefix workload.

## Why this matters

Most LLM prompts begin with thousands of tokens shared with earlier requests, such as a system prompt, tool definitions, a retrieved document, or conversation history. A replica that already holds those tokens in its KV cache skips prefilling them, cutting most of the time to first token.

A load-only balancer spreads each prefix across all replicas. Each replica sees any given prefix less often and evicts it before it returns, so adding replicas lowers the cache-hit rate. Pinning every prefix strictly to one replica creates the opposite failure. The single most popular prefix piles onto one overloaded replica while the rest sit idle. Production LLM routers balance these two extremes so hot prefixes spill over just enough to keep queues even.
