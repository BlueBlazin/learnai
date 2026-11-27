A semantic cache answers a request without calling an LLM when a similar request was answered recently. It stores each response under the embedding of its question and looks up new questions by cosine similarity. Implement `namespace_of` and `SemanticCache` to scope lookups strictly, return the best live match, enforce fixed lifetimes, and evict least recently used entries when full.

## Specification

```python
def namespace_of(request: dict) -> tuple | None: ...


class SemanticCache:
    def __init__(self, threshold: float, ttl_s: float, capacity: int): ...
    def get(self, embedding: list[float], now: float, namespace) -> str | None: ...
    def put(self, embedding: list[float], response: str, now: float, namespace) -> None: ...
```

### The namespace of a request

Two requests may share a cached response only if every setting that shapes the output, other than the user question, is identical. `namespace_of(request)` extracts that exact scope or signals that the request must not be cached.

`request` is a dict containing:

- `"model"`: a string naming the model.
- `"messages"`: a list of `{"role": ..., "content": ...}` dicts with string contents.
- `"temperature"` (optional): a number $\ge 0$. If absent, it defaults to `1.0`.
- `"max_tokens"` (optional): an int. If absent, it defaults to `None`.
- `"tools"` (optional): a list of tool definitions. If absent or empty, the request uses no tools.

Ignore any other keys in `request`.

`namespace_of(request)` returns:

- `None` if the request has at least one tool or if its temperature is not `0`. Tool-using responses depend on live external state, and sampled responses are meant to vary across calls.
- Otherwise, the tuple `(model, system, max_tokens)`, where `system` is the `"content"` of every message with role `"system"`, in order, joined with `"\n"` (or `""` if there are no system messages).

### The cache

An **entry** stores a `(namespace, embedding, response)` triple along with the timestamp $t_0$ when it was stored. The cache holds at most `capacity` entries across all namespaces combined.

- **Similarity.** For two embeddings $a$ and $b$, similarity is their cosine similarity computed in Python floats:
  $$\operatorname{sim}(a, b) = \frac{a \cdot b}{\lVert a \rVert_2 \, \lVert b \rVert_2}$$
  All embeddings are non-zero lists of floats of the same length, and they are not necessarily unit-normalized.
- **Expiry.** An entry stored at time $t_0$ is *live* at time `now` while `now - t0 < ttl_s`, and *expired* once `now - t0 >= ttl_s`. An entry's expiration time is fixed when it is stored and is never extended by hits.
- **Recency.** An entry is *used* when it is stored by `put` and whenever `get` returns its response. The least recently used entry is the one whose last use occurred earliest in the call sequence.

**`get(embedding, now, namespace)`** considers only entries that satisfy all three conditions:

1. Their namespace equals `namespace` (using `==`).
2. They are live at `now`.
3. $\operatorname{sim}(\text{embedding}, \text{entry embedding}) \ge \texttt{threshold}$.

If no entry qualifies, return `None` without modifying the cache. Otherwise, return the response of the qualifying entry with the **highest similarity**. Break ties in favor of the entry **stored most recently**. Mark the returned entry as the most recently used.

**`put(embedding, response, now, namespace)`** performs three steps in order:

1. Remove every entry across all namespaces that is expired at `now`.
2. If the cache still holds `capacity` entries, remove the least recently used entry.
3. Store the new entry and mark it as the most recently used.

`put` always inserts a new entry, even if an identical embedding is already stored. The tie-breaking rule in `get` ensures the newer response is preferred. Timestamps passed as `now` never decrease from one call to the next.

## Example

Let `cache = SemanticCache(threshold=0.8, ttl_s=60, capacity=2)`, with two distinct namespaces `ns` and `other`:

| call | result | why |
|---|---|---|
| `put([1, 0], "A", 0, ns)` | | stores A at $t_0 = 0$ |
| `put([3, 4], "B", 10, ns)` | | stores B at $t_0 = 10$ |
| `get([4, 3], 20, ns)` | `"B"` | similarities are `0.8` (A) and `0.96` (B), so both pass `0.8` and the higher similarity wins |
| `get([4, 3], 20, other)` | `None` | no entries exist in namespace `other` |
| `get([1, 0], 59, ns)` | `"A"` | A has similarity `1.0` and B has `0.6` |
| `get([1, 0], 60, ns)` | `None` | A expired at `60` (`60 - 0 < 60` is false) even though it was hit at `59`, and B is below `0.8` |
| `put([0, 1], "C", 65, ns)` | | expired entry A is purged, leaving room for B and C |
| `get([3, 4], 65, ns)` | `"B"` | B has similarity `1.0` and C has `0.8`, making B most recently used |
| `put([-1, 0], "D", 66, ns)` | | neither B nor C is expired and capacity is full, so C (least recently used) is evicted |
| `get([0, 1], 67, ns)` | `"B"` | C was evicted, and B matches with similarity `0.8` |

Example calls to `namespace_of`:

```python
request = {
    "model": "small",
    "temperature": 0,
    "messages": [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "What is the capital of Peru?"},
    ],
}
namespace_of(request)  # ("small", "Answer in French.", None)
namespace_of({**request, "temperature": 0.7})  # None (sampled)
namespace_of({**request, "tools": [{"type": "function", ...}]})  # None (uses tools)
```

## Constraints

- `capacity` $\ge 1$, `ttl_s` $> 0$, and $-1 \le \texttt{threshold} \le 1$.
- Tests use at most a few hundred entries and a few thousand calls, so a linear scan in `get` is expected.
- Responses are compared for exact string equality.
- Whenever a test relies on a similarity equaling `threshold` or another similarity, the comparison is exact in floating point (identical embeddings, or exact values such as `0.6`, `0.8`, `1.0` from vectors like `[3, 4]` and `[1, 0]`). All other similarities differ from each other and from `threshold` by at least `1e-9`.
- Timestamps are multiples of `1/64` s, so `now - t0` is exact in floating point.
- One test drives the cache with a simulated clock. Requests arrive over time, misses invoke a simulated model that takes `2` s before storing the response, and requests whose namespace is `None` bypass the cache. Your hit/miss sequence must match the reference exactly, and no request may receive a response generated for a different question, model, or system prompt.

## Why this matters

Semantic caches sit in front of LLM gateways and support bots where users frequently ask the same questions in slightly different wording. A cache hit replaces seconds of generation with a millisecond vector lookup.

However, false hits are far worse than misses. Keying on the question alone leaks responses across models, system prompts, or tenants. Returning the first match above the threshold can serve a nearby question when an exact match sits right behind it. Extending an entry's lifetime on every hit keeps popular stale answers alive forever, and caching tool-using calls freezes outdated real-world data.
