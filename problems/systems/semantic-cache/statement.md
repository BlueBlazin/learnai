A semantic cache answers a request without calling the model when a *similar*
request was answered before: it stores each response under the embedding of its
question and looks new questions up by cosine similarity. A hit saves a model
call. A **false hit** returns somebody else's answer, so the lookup has to be
strict about four things: which entries it may consider at all, which of several
matches it returns, how long an entry lives, and which entry goes when the cache is
full.

## Specification

```python
def namespace_of(request: dict) -> tuple | None: ...


class SemanticCache:
    def __init__(self, threshold: float, ttl_s: float, capacity: int): ...
    def get(self, embedding: list[float], now: float, namespace) -> str | None: ...
    def put(self, embedding: list[float], response: str, now: float, namespace) -> None: ...
```

### The namespace of a request

Two requests may share a cached response only if everything that shapes the
response, other than the question, is the same. `namespace_of` reduces a chat
request to that part, or says that the request must not be cached at all.

`request` is a dict with the keys

- `"model"`: a string;
- `"messages"`: a list of `{"role": ..., "content": ...}` dicts with string contents;
- optionally `"temperature"` (a number $\ge 0$; **absent means 1.0**, the API's
  default), `"max_tokens"` (an int; absent means `None`), and `"tools"` (a list;
  absent or empty means no tools). Other keys may be present and are ignored.

`namespace_of(request)` returns

- `None` if the request has at least one tool, or if its temperature is not 0: a
  tool-using response depends on the state of the world, and a sampled response
  is supposed to differ from call to call;
- otherwise the tuple `(model, system, max_tokens)`, where `system` is the contents
  of all messages with role `"system"`, in order, joined with `"\n"` (the empty
  string if there are none).

### The cache

An **entry** is a stored `(namespace, embedding, response)` together with the time
it was stored. The cache holds at most `capacity` entries, over all namespaces.

- **Similarity** of two embeddings is their cosine,
  $\operatorname{sim}(a, b) = \dfrac{a \cdot b}{\lVert a \rVert_2 \, \lVert b \rVert_2}$,
  computed in Python floats. Embeddings are lists of floats of one common length,
  never zero, and not necessarily of unit length.
- **Expiry.** An entry stored at time $t_0$ is *live* at time `now` while
  `now - t0 < ttl_s`, and *expired* from then on. Nothing ever extends an entry's
  life: its expiry is fixed when it is stored.
- **Recency.** An entry is *used* when it is stored and each time `get` returns its
  response. The least recently used entry is the one whose last use lies furthest
  back in the sequence of calls.

**`get(embedding, now, namespace)`** considers the entries that

1. have exactly this `namespace` (compared with `==`),
2. are live at `now`, and
3. have $\operatorname{sim}(\text{embedding}, \text{entry}) \ge$ `threshold`.

If there is none, it returns `None` and changes nothing. Otherwise it returns the
response of the entry with the **highest similarity**; among entries with equal
similarity, the one **stored most recently**. That entry becomes the most recently
used.

**`put(embedding, response, now, namespace)`**, in this order:

1. removes every entry that is expired at `now`, in every namespace;
2. if the cache still holds `capacity` entries, removes the least recently used one;
3. stores the new entry, which becomes the most recently used.

`put` always adds an entry, also when an equal embedding is already stored; the
tie rule of `get` then prefers the newer response. `now` never decreases from one
call to the next.

## Example

`cache = SemanticCache(threshold=0.8, ttl_s=60, capacity=2)`, with `ns` and `other`
two different namespaces:

| call | result | why |
|---|---|---|
| `put([1, 0], "A", 0, ns)` | | |
| `put([3, 4], "B", 10, ns)` | | |
| `get([4, 3], 20, ns)` | `"B"` | similarities `0.8` (A) and `0.96` (B): both reach the threshold, the higher one wins, not the first one |
| `get([4, 3], 20, other)` | `None` | no entry in that namespace |
| `get([1, 0], 59, ns)` | `"A"` | similarity `1.0`; B has `0.6` |
| `get([1, 0], 60, ns)` | `None` | A is expired (`60 - 0 < 60` is false), although it was used at 59; B is below the threshold |
| `put([0, 1], "C", 65, ns)` | | A is expired and removed; B and C are stored |
| `get([3, 4], 65, ns)` | `"B"` | similarity `1.0`; C has `0.8` |
| `put([-1, 0], "D", 66, ns)` | | nothing has expired and the cache is full: C is the least recently used (B was used at 65) and is removed |
| `get([0, 1], 67, ns)` | `"B"` | similarity `0.8` with B; C is gone |

And the namespace:

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
namespace_of({**request, "temperature": 0.7})  # None: sampled
namespace_of({**request, "tools": [{"type": "function", ...}]})  # None: uses tools
```

## Constraints

- `capacity` $\ge 1$, `ttl_s` $> 0$, $-1 \le$ `threshold` $\le 1$. At most a few
  hundred entries and a few thousand calls; a linear scan per `get` is expected.
- Responses are compared exactly. Where the tests rely on a similarity being equal
  to the threshold or to another similarity, it is exact in floating point: the two
  entries have the same embedding, or the numbers are those of the example
  (`0.6`, `0.8`, `1.0` from vectors such as `[3, 4]` and `[1, 0]`). All other
  similarities are at least `1e-9` apart from each other and from the threshold.
- Times are multiples of 1/64 s, so `now - t0` is exact.
- One case drives the cache from a simulated clock: requests arrive over time,
  a miss calls a model that takes 2 s and stores its answer when it arrives, and
  requests whose namespace is `None` bypass the cache. The sequence of hits and
  misses must equal the reference's, and no request may receive an answer that was
  generated for a different question, model, or system prompt.

## Why this matters

Semantic caches sit in front of LLM APIs in gateways and support bots, where a
large share of the traffic is the same few questions in different words, and a hit
replaces seconds of generation with a vector lookup. The savings are real and so
are the failures. A cache keyed on the question alone serves the answer of one
model, system prompt, or tenant to another. "First match above the threshold"
returns a neighbor when the exact question is stored right behind it. An entry
whose life is extended by every hit never expires as long as it is popular, which
is exactly when a stale answer does the most damage. And caching a tool-using
request freezes yesterday's weather.
