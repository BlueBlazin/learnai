Write the retry loop of an inference client: when the server says it is overloaded,
wait and try again, in a way that helps the server recover instead of keeping it
down. A retry loop is ten lines. Each of the ten has a wrong version that looks
reasonable in a test with one client and causes an outage with a thousand.

![This problem is the Client, retrying when the server is overloaded.](assets/map.svg)

## Specification

```python
@dataclass(frozen=True)
class RetryPolicy:  # given in the starter
    max_attempts: int = 4  # sends in total, the first one included; >= 1
    base_s: float = 0.5  # backoff ceiling before the first retry
    cap_s: float = 8.0  # the ceiling never exceeds this
    deadline_s: float = 30.0  # total time budget of the call
    idempotent: bool = True  # may the request be resent when its outcome is unknown?


async def call_with_retry(send, policy: RetryPolicy, uniforms: list[float], budget=None): ...
```

`send` is an async callable without arguments. Each `await send()` is one attempt: it
transmits the request and either returns a **response**, an object with an int
`.status` and a dict `.headers` (names and values are strings), or raises
`ConnectionError` (the connection broke; whether the server executed the request is
unknown). An attempt takes time.

Time is the running event loop's clock: `asyncio.get_running_loop().time()` and
`asyncio.sleep`. Under the judge it is simulated. Do not use the real clock, and do
not draw random numbers: the jitter comes from `uniforms`.

`call_with_retry` sends the request, and after each attempt decides whether to send
it again. It **retries** only if all of the following hold, and otherwise ends.

1. **The failure is retryable.**
   - A response with status **429** or **503** is retryable: the server refused the
     request because it is overloaded and did not execute it.
   - A `ConnectionError` is retryable only if `policy.idempotent` is true.
   - Nothing else is: not a success, and not any other status (400, 404, 500, ...).
2. **Attempts remain**: fewer than `policy.max_attempts` attempts have been made.
3. **The budget allows it**: `budget` is `None`, or `budget.allows_retry()` is true
   (see below).
4. **The retry starts before the deadline.** With $t_0$ the time `call_with_retry`
   was called, $t$ the current time, and $d$ the delay computed below, the retry is
   made only if $(t - t_0) + d < \text{deadline\_s}$. An attempt in flight is never
   interrupted; the deadline only stops new attempts.

When it retries, it waits $d$ seconds (`asyncio.sleep(d)`) and sends again. When it
ends, it returns the last response, whatever its status, or raises the last
`ConnectionError`.

**The delay $d$**, computed after conditions 1 to 3 hold:

- **`Retry-After`.** If the response has a header whose name equals `retry-after`
  ignoring case, and whose value, stripped of surrounding whitespace, consists only
  of ASCII digits, then $d$ is that number of seconds, exactly: no jitter, no cap. A
  value of any other form (an HTTP-date, a negative or fractional number, an empty
  string) is treated as if the header were absent.
- **Backoff with full jitter**, otherwise (always after a `ConnectionError`). With
  $k$ the number of retries this call has already made, however their delays were
  chosen, and $u$ the next unused element of `uniforms`:

$$
d = u \cdot \min\!\big(\text{cap\_s},\ \text{base\_s} \cdot 2^{k}\big).
$$

  The first backoff delay of a call uses `uniforms[0]`, the next `uniforms[1]`, and
  so on. A delay taken from `Retry-After` consumes no uniform. `uniforms` is long
  enough.

**The retry budget.** `budget` is `None` or an object shared by many calls, with
three methods: `success()`, `failure()`, and `allows_retry() -> bool`. After every
attempt, before deciding anything else:

- call `budget.failure()` once if the attempt ended in status 429 or 503 or in a
  `ConnectionError` (retryable or not);
- call `budget.success()` once if it returned a status below 400;
- call neither for any other status.

Then condition 3 asks `budget.allows_retry()`. The budget the tests use is a bucket of
10 tokens: `failure()` removes one (not below 0), `success()` adds 0.5 (not above
10), and `allows_retry()` is true while more than 5 remain.

## Example

```python
policy = RetryPolicy(max_attempts=4, base_s=0.5, cap_s=8.0, deadline_s=30.0)
```

The call starts at $t = 1.0$ with `uniforms = [0.5, 0.75, 0.5]`; the server answers
503, 503, 200, each at once.

| time | event |
|---|---|
| 1.0 | attempt 1: 503. $k = 0$: $d = 0.5 \cdot \min(8, 0.5) = 0.25$ |
| 1.25 | attempt 2: 503. $k = 1$: $d = 0.75 \cdot \min(8, 1.0) = 0.75$ |
| 2.0 | attempt 3: 200, returned |

More cases, each starting at $t = 0$ with every uniform equal to 0.5:

- A server that always answers 503: sends at 0, 0.25, 0.75, 1.75, then the fourth 503
  is returned (`max_attempts` is 4).
- A 404: one send; the 404 is returned.
- A 429 with `Retry-After: 2`, then a 503 without the header, then 200: sends at 0,
  2.0, and 2.5. The second delay is a backoff with $k = 1$, since one retry has been
  made, and it uses `uniforms[0]`, since the `Retry-After` delay consumed none:
  $0.5 \cdot \min(8, 1.0) = 0.5$.
- `base_s = 1`, `cap_s = 4`, `max_attempts = 7`, always 503: the ceilings are 1, 2, 4,
  4, 4, 4, so the sends are at 0, 0.5, 1.5, 3.5, 5.5, 7.5, 9.5.
- `base_s = 1`, `deadline_s = 4`, `max_attempts = 10`, always 503: sends at 0, 0.5,
  1.5, 3.5. The next delay would be 4.0, and $3.5 + 4.0 \ge 4$, so the call returns
  the 503 at 3.5 without waiting.
- The connection breaks 0.5 s after the first send: with `idempotent=True` the retry
  is sent at $0.5 + 0.25 = 0.75$; with `idempotent=False` the `ConnectionError` is
  raised at 0.5.

## Constraints

- `1 <= max_attempts <= 20`; `0 < base_s <= cap_s`; `deadline_s > 0`; every uniform
  is in $[0, 1)$.
- Send times are compared with an absolute tolerance of `1e-9` s; the number of sends
  and the outcome exactly.
- **Thundering herd.** 64 clients call at $t = 0$ against a server that answers 503
  until $t = 3$ (`base_s = 2`, `cap_s = 16`, `max_attempts = 6`, an attempt takes
  1/64 s). No window of 1/8 s after $t = 0$ may contain more than **24** sends. The
  reference has at most 12; without jitter all 64 retries land in one instant.
- **Retry storm.** 40 calls share one budget; the first 20 meet a server that always
  answers 503. The 40 calls may send at most **50** requests in total. The reference
  sends 44; without the budget it is 101.
- Simultaneous events: a retry whose delay is 0 is sent in the same instant as the
  response that caused it. Nothing else in the tests depends on the order of
  simultaneous events.

## Why this matters

An overloaded server recovers when load drops. Clients that retry at once, or all
together, make sure it does not: every rejected request comes back, the 429s cost
the server work too, and a brief overload becomes a sustained one. The pieces of
this function each remove one such mechanism. Exponential backoff reduces the
retry rate as the trouble persists; jitter spreads clients that failed at the same
moment so that they do not return at the same moment; `Retry-After` lets the server,
which knows its queue, set the pace (the `retry_after_s` of `admission-control` is
what it sends); the attempt limit and the deadline bound the work spent on a request
nobody is waiting for any longer; and the budget bounds the *fleet's* retries, so
that during a full outage the load on the server stays close to normal instead of
rising to `max_attempts` times normal. Retrying the wrong things is the other half: a 400 will
never succeed, and a non-idempotent request whose connection dropped may already
have been executed, so sending it again runs a generation, or a payment, twice.
