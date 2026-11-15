## The idea

An exact cache is a dictionary: the key either matches or it does not. A semantic
cache replaces the key with an embedding and "matches" with "is similar enough",
and that one change creates every question this problem asks. Similarity is a
matter of degree, so several entries can match and one has to be chosen. Similar
questions under different instructions have different answers, so something
other than the embedding has to stay an exact key. And a cached answer was true
when it was generated, so it needs an expiry that popularity cannot postpone.

**Scope first, then similarity.** The response to a request is a function of the
question *and* of the model, the system prompt, and the generation settings.
Only the question is fuzzy. Everything else goes into a namespace that is
compared exactly, and a lookup never looks outside its namespace. Two kinds of
request have no namespace at all: a request with tools (its answer depends on
what the tools return today) and a sampled request (the caller asked for
variety; at temperature 0 the remaining settings that change the output are the
model, the prompt, and the length limit).

**Filter, then choose.** `get` first discards what it may not return: other
namespaces, expired entries, similarities below the threshold. Among what is left
it returns the *best*, not the first. The order of the two steps is not a detail:
choosing the global best and then checking whether it is expired turns a live
match into a miss whenever a stale entry happens to be closer.

**Two clocks per entry.** The time an entry was stored decides when it expires,
and nothing moves it. The time it was last used decides who is evicted when the
cache is full, and every hit moves it. Mixing them up in one direction gives
entries that never expire; in the other, a cache that evicts its most useful
entry.

**Make room in the right order.** A full cache first drops what is already dead
and only then evicts something alive.

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

`eq=False` keeps `list.remove` comparing entries by identity; with the generated
`__eq__`, two entries with equal fields would be interchangeable and the wrong one
could be moved.

## Common mistakes

- **Returning the first match above the threshold.** With a threshold of 0.8, a
  neighbor at 0.8 that was stored earlier wins over the identical question at
  1.0. The loop needs a running maximum, not an early `return`.
- **Hits across namespaces.** Keying on the embedding alone returns the answer
  of another model or another system prompt. The embedding of "What is the
  capital of Peru?" is the same whether the system prompt says "Answer in
  French" or not.
- **Caching what must not be cached.** Treating a missing temperature as 0, or
  ignoring `tools`, gives sampled and tool-using requests a namespace.
- **Refreshing the TTL on a hit** (sliding expiration). The most popular entry
  then lives forever. Session caches want that; an answer cache does not.
- **Choosing before filtering.** Taking the most similar entry and then finding
  it expired (or in another namespace) reports a miss although a live match
  exists.
- **Not updating recency on a hit**, which turns LRU into FIFO and evicts the
  entry that is saving the most calls.
- **Evicting before purging.** A full cache with dead entries in it evicts a live
  one.
- **A capacity per namespace.** The bound is on memory, and memory does not care
  how many system prompts there are.
- **Off by one at the boundaries**: `>` for the threshold (an exact repeat of a
  stored question has similarity 1.0 and must hit with `threshold=1.0`), `<=` for
  the lifetime.
- **Ties to the older entry.** After `put` of an updated answer under the same
  embedding, the old answer keeps being served until it expires.
- **A dot product instead of a cosine.** It ranks long vectors first and makes the
  threshold meaningless unless every embedding is normalized.

## Complexity

`get` is a linear scan: $O(n \cdot d)$ for $n$ entries of dimension $d$; `put` is
$O(n)$. That is fine for a few thousand entries. Beyond that the scan becomes a
vector index (the earlier problems of this track) that returns the top $k$
candidates, and the filter on namespace and expiry is applied to those, with $k$
large enough that filtering does not empty the result.

## In the wild

- GPTCache, the open-source semantic cache for LLM applications, documents this
  decomposition: an embedding generator, a vector store that "helps find the K
  most similar requests", a cache storage for the responses, a similarity
  evaluator that decides whether a candidate really matches, and a cache manager
  with LRU and FIFO eviction that counts entries (the documentation notes that
  eviction is by number of rows, not by memory). Its `Config` has a
  `similarity_threshold`, and its temperature parameter controls how often the
  cache is skipped for sampled requests.
- The "completion cache" is one of the cost-reduction strategies listed in
  *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving
  Performance* (arXiv 2305.05176): store responses and reuse them when a similar
  query arrives.
- Exact-match response caches and provider-side prompt caches avoid the false-hit
  problem by construction; a semantic cache trades that safety for hit rate,
  which is why the threshold and the namespace deserve the care.
