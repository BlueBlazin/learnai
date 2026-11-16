Implement the gate in front of an inference server's queue: admit a request only while
the queue stays within a bound on requests and a bound on prompt tokens, and tell a
rejected client when it is worth coming back. A server without this gate never says
no; under overload its queue, and with it everyone's latency, grows without limit.

## Specification

```python
@dataclass(frozen=True)
class Admitted: ...

@dataclass(frozen=True)
class Rejected:
    retry_after_s: float | None

class Admission:
    def __init__(self, max_queued_requests: int, max_queued_tokens: int,
                 drain_tokens_per_s: float): ...
    def admit(self, req) -> Admitted | Rejected: ...
    def start(self, req) -> None: ...
    def finish(self, req) -> None: ...
    queued_requests: int      # read-only properties
    queued_tokens: int
    running_requests: int
```

`Admitted` and `Rejected` are given in the starter. A request `req` is an object with
a unique hashable `id` and an int `prompt_tokens >= 1`; the tests pass
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects from the kit.

**The life of a request.** The server calls `admit(req)` when a request arrives. If
the answer is `Admitted`, the request is *queued*. Later the server calls `start(req)`
when it begins to run the request, which is then *running* and no longer queued, and
`finish(req)` when the request has completed. A rejected request is forgotten; the
same request may be offered to `admit` again later.

**The bounds** apply to queued requests only. Write $Q$ for the number of queued
requests and $T$ for the sum of their `prompt_tokens`. `admit(req)` with
$p$ = `req.prompt_tokens` returns

- `Admitted()` if $Q + 1 \le$ `max_queued_requests` **and** $T + p \le$
  `max_queued_tokens`; the request joins the end of the queue;
- `Rejected(retry_after_s)` otherwise; nothing changes.

Running requests count against neither bound: their place in the queue and their
tokens are released by `start`.

**`retry_after_s`** estimates how long the client should wait before the same request
would be admitted, assuming that nothing else arrives, that queued requests leave the
queue oldest first, and that the queue drains at `drain_tokens_per_s` prompt tokens
per second. Let the queued requests be $q_1, \dots, q_Q$, oldest first, with prompt
tokens $p_1, \dots, p_Q$.

- If $p >$ `max_queued_tokens`, the request can never be admitted, and
  `retry_after_s` is `None`.
- Otherwise let $j$ be the smallest number of departures after which the request fits:
  the smallest $j \ge 1$ with $(Q - j) + 1 \le$ `max_queued_requests` and
  $(T - p_1 - \dots - p_j) + p \le$ `max_queued_tokens`. Then

$$
\text{retry\_after\_s} = \frac{p_1 + \dots + p_j}{\text{drain\_tokens\_per\_s}} .
$$

  Such a $j$ always exists, because the request fits into an empty queue.

**`start(req)`** may be called for any queued request, not only the oldest (a
scheduler may reorder). "Oldest first" in the estimate refers to the requests that are
queued at the time of the `admit` call, in the order in which they were admitted.

**The properties** report the current state: `queued_requests` is $Q$,
`queued_tokens` is $T$, and `running_requests` is the number of requests between
`start` and `finish`.

The tests call `start` only for a queued request and `finish` only for a running one,
and never offer a request whose `id` is currently queued or running. The class reads
no clock and does not sleep; every method returns immediately.

## Example

`Admission(max_queued_requests=3, max_queued_tokens=1000, drain_tokens_per_s=500)`:

| call | result | queue after (tokens) | why |
|---|---|---|---|
| `admit(a)`, 400 tokens | `Admitted()` | a (400) | |
| `admit(b)`, 300 tokens | `Admitted()` | a, b (700) | |
| `admit(c)`, 400 tokens | `Rejected(0.8)` | a, b (700) | 700 + 400 > 1000; fits once a has left: 400 / 500 |
| `admit(d)`, 200 tokens | `Admitted()` | a, b, d (900) | |
| `admit(e)`, 50 tokens | `Rejected(0.8)` | a, b, d (900) | a fourth request; fits once a has left |
| `admit(f)`, 700 tokens | `Rejected(1.4)` | a, b, d (900) | after a: 500 + 700 > 1000; after a and b: 200 + 700 fits; (400 + 300) / 500 |
| `admit(g)`, 1200 tokens | `Rejected(None)` | a, b, d (900) | 1200 > 1000: never |
| `start(a)` | | b, d (500) | a is running: `running_requests` is 1 |
| `admit(c)`, 400 tokens | `Admitted()` | b, d, c (900) | |
| `finish(a)` | | b, d, c (900) | `running_requests` is 0 |

## Constraints

- `1 <= max_queued_requests <= 1000`, `1 <= max_queued_tokens <= 10^6`,
  `drain_tokens_per_s > 0`; at most a few hundred requests per test.
- Decisions are compared exactly; `retry_after_s` within `1e-9` (the tests use drain
  rates and token counts for which the quotient is exact).
- **The simulation.** Besides hand-written call sequences, the tests put the object in
  front of a simulated server with one slot that runs a request for
  `prompt_tokens / 1024` seconds, with `drain_tokens_per_s = 1024`. When the slot is
  free the server starts the oldest queued request (one case starts the shortest
  prompt instead). At one instant, a completion and the start that follows it are
  delivered before the arrivals of that instant. Arrivals come at up to three times
  the rate the server can sustain.
- In the simulation the tests check that the number of requests and of prompt tokens
  really waiting never exceed the bounds, that the three properties agree with what is
  really queued and running after every call, that every decision is the specified
  one, and, for the first-come-first-served server, that every admitted request
  finishes within
  `(max_queued_tokens + longest prompt) / 1024` seconds of its arrival: the running
  request takes at most one longest prompt, and everything queued up to and including
  the new request is at most `max_queued_tokens`.

## Why this matters

An inference server's throughput is fixed by its hardware, so when requests arrive
faster than they finish, the only thing that can grow is the queue. An unbounded
queue turns overload into latency: every request is eventually served, minutes late,
long after its client has timed out and retried, which adds load. A bounded queue
turns overload into fast, explicit rejections (HTTP 429 or 503 with a `Retry-After`
header) that a load balancer can route elsewhere, and keeps the latency of the
requests that are admitted within a limit that follows from the bounds. The token
bound matters as much as the request bound: ten queued requests can mean ten seconds
of prefill or ten minutes, depending on their prompts. And counting running requests
against the queue bound is the quiet version of the opposite failure: the server
rejects traffic while its queue is empty.
