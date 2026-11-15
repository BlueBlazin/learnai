Route each request to the replica that most likely still has the beginning of its
prompt in its KV cache, so that the shared system prompt is not prefilled again,
without letting a popular prefix pile every request onto one replica. The router
cannot see the replicas' caches: it keeps its own record of what it sent where, and
it has to decide when a match is worth following and when load wins.

## Specification

```python
class Router:
    def __init__(self, match_threshold: float, imbalance: int, index_size: int): ...
    def route(self, tokens: tuple[int, ...], replicas: list) -> int: ...
```

`route` is called once per request, in arrival order. `tokens` is the prompt as
token ids (at least one). `replicas` is a non-empty list of the replicas that can
take the request now, in no particular order; each has an int `id` (unique) and an
int `queue_len`, the number of requests it holds, waiting or running. `route`
returns the `id` of the replica the request goes to, a Python `int`.

**The index.** For every replica id the router remembers the last `index_size`
prompts it routed there, oldest forgotten first. Routing a prompt to a replica
always appends it, also when an identical prompt is already remembered (which then
counts twice toward `index_size`). The index starts empty, and a replica id never
seen before has an empty record.

**The match** of a prompt with a replica is the length of the longest common prefix
between `tokens` and any prompt remembered for that replica: the number of leading
positions at which the two agree, stopping at the first position where they differ
or where either ends. It is 0 for an empty record.

**The decision.** Let `least` be the smallest `queue_len` among `replicas`.

1. A replica is **eligible** if `queue_len - least <= imbalance`.
2. The **best** replica is the eligible one with the largest match; among equal
   matches the smaller `queue_len`; then the smaller `id`.
3. If `match(best) >= match_threshold * len(tokens)`, the request goes to `best`.
4. Otherwise it goes to the least loaded replica: the smallest `queue_len` among
   all `replicas`, then the smaller `id`.
5. The prompt is appended to the record of the chosen replica.

Only the replicas passed to this call are considered; records of other ids are
kept, in case those replicas are offered again.

**In the simulation**, the router stands in front of the kit's `SimCluster`: four
replicas, each a continuous-batching server that runs at most 8 sequences at once,
first come, first served; a step costs $2^{-6}$ s plus $2^{-12}$ s per prefilled
token plus $2^{-9}$ s per decoded sequence. Each replica has a prefix cache of
1200 tokens: when a request is admitted, the longest prefix it shares with a prompt
that replica has finished prefilling (in whole blocks of 16 tokens) is not prefilled
again, and the least recently used prompts are evicted beyond 1200 tokens, about
three prompts. The router is constructed as `Router(0.25, 4, 3)`. It must not read
anything from a replica except `id` and `queue_len`.

## Example

`Router(match_threshold=0.5, imbalance=2, index_size=2)` and three replicas. Queue
lengths are written `{id: queue_len}`.

| call | `tokens` | queues | returns | why |
|---|---|---|---|---|
| 1 | `(1, 2, 3, 4, 5, 6, 7, 8)` | `{0: 0, 1: 0, 2: 0}` | 0 | nothing is remembered: least loaded, lowest id |
| 2 | `(1, 2, 3, 4, 5, 6, 9, 9)` | `{0: 1, 1: 0, 2: 0}` | 0 | replica 0 matches 6 of 8 tokens ($6 \ge 0.5 \cdot 8$) and is 1 above the least loaded |
| 3 | `(1, 2, 3, 7, 7, 7, 7, 7)` | `{0: 2, 1: 0, 2: 0}` | 1 | the best match is 3 tokens, below 4: least loaded, lowest id |
| 4 | `(1, 2, 3, 4, 5, 6, 7, 7)` | `{0: 3, 1: 1, 2: 0}` | 2 | replica 0 matches 7 tokens but is 3 above the least loaded: not eligible. Of the eligible ones, replica 1 matches 3: least loaded |
| 5 | `(1, 2, 3, 4, 5, 6, 7, 0)` | `{0: 2, 1: 1, 2: 1}` | 2 | replicas 0 and 2 both match 7 tokens; replica 2 has the shorter queue |

After call 4 the prefix `1, 2, 3, 4, 5, 6, 7` is remembered for two replicas, and
call 5 uses whichever of them is less busy: this is how a prefix that is too popular
for one replica spreads to a second one and no further than needed.

## Constraints

- `0 <= match_threshold <= 1`, `imbalance >= 0`, `index_size >= 1`; up to 16
  replicas; prompts of up to 4096 tokens. In the direct tests thresholds are
  multiples of 0.25, so `match_threshold * len(tokens)` is exact.
- The rules determine every decision. Decisions are compared exactly, on scripted
  calls and, in the simulation, on seeded workloads of 300 to 600 requests whose
  prompts consist of one of several shared prefixes (64 to 512 tokens) and a
  suffix of their own.
- **Against a load-only router** (least `queue_len`, lowest id on ties), on the
  same workload and cluster, with 600 requests over 8 prefixes of 256 to 512 tokens:

  | workload | cache-hit tokens | p95 time to first token |
  |---|---|---|
  | prefixes equally likely, 30 requests/s | at least 1.8 times the baseline's | at most 0.6 times the baseline's |
  | one hot prefix (weights $(g+1)^{-2}$), 40 requests/s | at least 1.1 times | at most 0.75 times |

  The reference reaches 2.9 times and 0.13 times on the first, 1.29 times and 0.45
  times on the second (there the load-only router already hits often, because the
  hot prefix is cached everywhere).
- In both, the replica chosen for a request is never more than `imbalance` = 4
  requests above the least loaded one at that moment. A router that always follows
  the best match sends requests to a replica 156 requests above the least loaded
  one on the hot-prefix workload.

## Why this matters

Most prompts start with the same few thousand tokens as some earlier prompt: a
system prompt, a tool list, a document, the conversation so far. A replica that has
those tokens' keys and values cached skips their prefill, which is most of the time
to first token. Behind a load balancer that only looks at load, each replica has
seen each prefix only a fraction of the time and has evicted it by the time it
comes again, so adding replicas lowers the cache-hit rate. Pinning each prefix to
one replica restores the hits and creates the opposite failure: the one prefix
everybody uses lands on a single replica, which queues while the rest idle. Every
production router for LLM traffic is some resolution of that tension, and the bug
reports are always one of the two extremes.
