## The idea

An exact cache behaves like a dictionary where keys either match or miss. A semantic cache replaces exact keys with embeddings and equality with a similarity threshold. That single change introduces four design requirements. Multiple entries can pass the threshold, so the cache must pick the highest similarity. Identical questions under different instructions need different answers, so non-question parameters must stay exact keys. Generated answers go stale over time, so expiration must remain fixed regardless of popularity. When the cache is full, expired entries must be purged before evicting the least recently used live entry.

**Scope first, then similarity.** A model's response depends on the question, the model ID, the system prompt, and generation settings. Only the question is matched fuzzily. Everything else belongs in an exact namespace, and lookups never cross namespace boundaries. Requests with tools cannot be cached because their answers depend on live external state. Sampled requests (`temperature != 0`) cannot be cached because callers expect variation. At `temperature == 0`, the remaining parameters that affect the output are the model, system prompt, and `max_tokens`.

**Filter, then choose.** `get` first discards entries from other namespaces, expired entries, and entries below `threshold`. Among the remaining candidates, it selects the highest similarity rather than the first match. Filtering before choosing is critical. Picking the globally closest entry first and then checking if it is expired turns a valid live hit into a miss whenever a stale entry happens to be closer.

**Two clocks per entry.** Each entry's creation timestamp (`stored_at`) governs expiration and never changes. Its recency position governs eviction when the cache is full and updates on every hit. Confusing the two either creates entries that never expire or evicts the most frequently used entries.

**Purge before evicting.** When `put` runs on a full cache, it first removes all expired entries across all namespaces. Only if the cache is still at capacity does it evict the least recently used live entry.

## Reference solution (Python)

```python
def namespace_of(request):
    if request.get("tools") or request.get("temperature", 1.0) != 0:
        return None
    system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
    return (request["model"], system, request.get("max_tokens"))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@dataclass(eq=False)
class Entry:
    namespace: object
    embedding: list[float]
    response: str
    stored_at: float
    order: int


class SemanticCache:
    def __init__(self, threshold, ttl_s, capacity):
        self.threshold, self.ttl_s, self.capacity = threshold, ttl_s, capacity
        self.entries = []  # least recently used first
        self.puts = 0

    def live(self, entry, now):
        return now - entry.stored_at < self.ttl_s

    def get(self, embedding, now, namespace):
        best, best_key = None, None
        for entry in self.entries:
            if entry.namespace != namespace or not self.live(entry, now):
                continue
            score = cosine(embedding, entry.embedding)
            if score < self.threshold:
                continue
            key = (score, entry.order)
            if best is None or key > best_key:
                best, best_key = entry, key
        if best is None:
            return None
        self.entries.remove(best)
        self.entries.append(best)
        return best.response

    def put(self, embedding, response, now, namespace):
        self.entries = [entry for entry in self.entries if self.live(entry, now)]
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append(Entry(namespace, list(embedding), response, now, self.puts))
        self.puts += 1
```

Setting `eq=False` makes `list.remove` compare entries by object identity. Without it, the default dataclass `__eq__` would treat two entries with identical fields as equal and could move the wrong one.

## Common mistakes

- **Returning the first match above `threshold`.** With `threshold=0.8`, an earlier neighbor at `0.8` beats an exact match at `1.0` stored behind it. Scan all live entries in the namespace and track the maximum.
- **Matching across namespaces.** Keying only on the embedding returns answers generated for a different model or system prompt. The embedding of "What is the capital of Peru?" is identical whether the system prompt asks for French or English.
- **Caching non-cacheable requests.** Treating an absent `temperature` as `0`, or ignoring `tools`, incorrectly assigns a namespace to sampled or tool-using requests.
- **Refreshing TTL on a hit (sliding expiration).** Extending an entry's lifetime whenever it is read lets popular entries live forever. Session caches want sliding expiration, while response caches need fixed lifetimes.
- **Choosing before filtering.** Selecting the most similar entry first and then discarding it for being expired or in the wrong namespace causes false misses when a valid live match exists.
- **Not updating recency on a hit.** Leaving hit entries in place turns LRU eviction into FIFO and evicts the cache's most useful entries.
- **Evicting before purging expired entries.** A full cache that evicts before dropping dead entries throws away a live entry while expired ones still occupy slots.
- **Enforcing capacity per namespace.** Memory limits apply to the whole cache, regardless of how many distinct system prompts exist.
- **Off-by-one boundary checks.** Using `>` for the similarity threshold causes an exact repeat (`1.0`) to miss when `threshold=1.0`. Using `<=` for `now - stored_at` keeps expired entries alive at the exact expiry boundary.
- **Breaking ties in favor of the older entry.** After `put` inserts a fresher answer for the same embedding, the stale answer keeps winning until it expires.
- **Using dot product instead of cosine similarity.** Unnormalized dot products favor longer vectors and make the similarity threshold meaningless.

## Complexity

For $n$ cached entries of dimension $d$, `get` runs a linear scan in $O(n \cdot d)$ time and `put` runs in $O(n)$ time. This is fast enough for a few thousand entries. At larger scales, the linear scan is replaced by a vector index that retrieves the top $k$ candidates, followed by namespace and expiry filtering with $k$ large enough to leave valid matches.

## In the wild

- GPTCache, an open-source semantic cache for LLM applications, splits the system into an embedding generator, a vector store that finds the $K$ most similar requests, a response store, a similarity evaluator, and a cache manager with LRU or FIFO eviction counted by number of rows. Its `Config` exposes a `similarity_threshold` and uses temperature to decide when to skip the cache on sampled requests.
- *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* (arXiv 2305.05176) highlights a completion cache as a core cost-reduction strategy for reusing responses across similar queries.
- Exact-match response caches and provider-side prompt caches avoid false hits by construction. A semantic cache trades that strict safety for higher hit rates, which is why careful namespace isolation and similarity thresholds are essential.
