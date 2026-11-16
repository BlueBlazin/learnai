Implement the step planner of a continuous-batching engine that works under a token
budget: every step may process at most `max_batch_tokens` tokens, running sequences
are decoded first, and prompts get what is left, in chunks. A long prompt is spread
over several steps and never makes the sequences that are already generating wait.

## Specification

```python
def plan(running, prefilling, waiting, max_batch_tokens, long_prefill_threshold):
    ...
    return prefill, decode
```

`plan` is a pure function. It is called once before every engine step with the state
of the server and returns the two arguments of `engine.step`.

| argument | meaning |
|---|---|
| `running` | ids of the sequences whose prompt is completely prefilled, in admission order. Each can decode one token in this step. |
| `prefilling` | `(id, remaining_tokens)` for every admitted sequence whose prompt is only partly prefilled, in admission order. `remaining_tokens >= 1`. |
| `waiting` | `(id, prompt_tokens)` for every request that has arrived and is not admitted, in arrival order. |
| `max_batch_tokens` | the budget of one step, `>= 1`: prompt tokens plus decoded sequences. |
| `long_prefill_threshold` | the most prompt tokens one sequence may receive in one step, `>= 1`. |

It returns the tuple `(prefill, decode)`: `prefill` is a list of `(id, n_tokens)`
chunks and `decode` is a list of ids. The rule:

1. **Decodes first.** `decode` is `running`, in order. Each decode costs one token of
   the budget. (If there are more running sequences than budget, `decode` is the first
   `max_batch_tokens` of them.)
2. **Then prompts, first come, first served.** Walk through `prefilling` and then
   `waiting`, in order. Each candidate gets a chunk of

   $$
   n = \min(\text{remaining tokens},\ \text{long\_prefill\_threshold},\ \text{budget left}),
   $$

   and the budget shrinks by $n$. Stop when the budget is used up. Candidates that are
   not reached get nothing in this step and do not appear in `prefill`.

So a prompt that does not fit is **split**: it takes what the budget and the threshold
allow now and continues in later steps. Nobody is skipped: while budget is left, the
next candidate in line gets at least one token. A sequence appears at most once in the
two lists. In particular, a sequence that receives its last chunk in this step is
*not* decoded in this step; that chunk produces its first output token, and the
sequence is in `running` at the next call.

**The loop around `plan`.** The tests run this loop, with a
[`SimEngine`](../../../python/src/learnai/kit/simengine.py) whose
`max_batch_tokens` is the budget. You do not write it.

```python
waiting = deque()                                   # requests, in arrival order
while True:
    if not waiting and not engine.live and queue.empty():
        waiting.append(await queue.get())           # idle
    while not queue.empty():
        waiting.append(queue.get_nowait())

    left = {i: engine.remaining_prefill(i) for i in engine.live}   # admission order
    running = [i for i, n in left.items() if n == 0]
    prefilling = [(i, n) for i, n in left.items() if n > 0]
    prefill, decode = plan(running, prefilling, [(r.id, r.prompt_tokens) for r in waiting],
                           max_batch_tokens, long_prefill_threshold)

    chunked = {i for i, _ in prefill}
    while waiting and waiting[0].id in chunked:     # a chunk admits a waiting request
        engine.admit(waiting.popleft())
    await engine.step(prefill=prefill, decode=decode)
```

Time is virtual and counted in **milliseconds**. A step over $P$ prompt tokens and $D$
decoded sequences takes $8 + 0.25\,P + D$ ms. A sequence produces its first output
token at the end of the step that completes its prompt and one token per step in which
it is decoded. There is no limit on the number of live sequences other than the
budget. The engine refuses, and the case fails, if a step exceeds the budget, is empty,
names a sequence twice, decodes a sequence whose prompt is incomplete, or gives a
sequence more prompt tokens than it has left.

A request that arrives during a step, or at the exact instant a step ends, is in
`waiting` at the next call.

## Example

`max_batch_tokens = 32`, `long_prefill_threshold = 24`. Requests as (arrival, prompt
tokens, output tokens): request 0 `(0, 16, 4)`, request 1 `(0, 40, 2)`, request 2
`(3, 8, 2)`.

| starts | `running` | `prefilling` | `waiting` | `prefill` | `decode` | takes |
|---|---|---|---|---|---|---|
| 0 | | | `(0, 16), (1, 40)` | `(0, 16), (1, 16)` | | 8 + 8 = 16 |
| 16 | `0` | `(1, 24)` | `(2, 8)` | `(1, 24), (2, 7)` | `0` | 8 + 7.75 + 1 = 16.75 |
| 32.75 | `0, 1` | `(2, 1)` | | `(2, 1)` | `0, 1` | 8 + 0.25 + 2 = 10.25 |
| 43 | `0, 2` | | | | `0, 2` | 10 |

At time 0 the budget is 32: request 0 takes its whole prompt (16), and request 1 is
capped by the 16 tokens that are left. At 16 the decode of request 0 costs one token,
leaving 31: request 1 is capped by the threshold (24), and request 2, which arrived at
3, gets the last 7. Request 1 completed its prompt in the second step and is decoded
from the third one on. The times to first token are 16, 32.75, and 40 ms.

## Constraints

- `1 <= max_batch_tokens <= 512`, `1 <= long_prefill_threshold <= 512`; prompts have
  1 to 4,000 tokens. Ids are opaque. Do not modify the argument lists.
- `plan` is deterministic, and the tests compare its output exactly: on direct calls
  with literal results, on 300 random states with the reference, and through the loop
  above on seeded Poisson workloads of 60 requests (literal steps on the small
  scenarios, the reference's steps on the large ones).
- **Stall bound.** Six sequences are generating when a 4,000-token prompt arrives,
  followed by three 24-token prompts. With `max_batch_tokens = 512` and
  `long_prefill_threshold = 512`, the longest gap between two consecutive tokens of
  any of the six must be at most **150 ms** (the largest possible step is
  8 + 0.25 · 506 + 6 = 140.5 ms; prefilling the prompt in one step would take over a
  second).
- **Head-of-line bound.** The same scenario with `long_prefill_threshold = 128`: each
  of the three short prompts must have its first token within **120 ms** of arriving,
  and the gap bound above still holds.

## Why this matters

Prefill and decode have opposite shapes. A prefill pushes thousands of tokens through
the model in one pass and takes hundreds of milliseconds; a decode step handles one
token per sequence and takes a few. Put a long prompt into a batch of decoding
sequences without a budget and every one of them freezes for the duration: users see
streaming text stop for a second whenever somebody else pastes a document. Running
prefills separately instead protects the stream and wastes throughput. Chunked prefill
with a token budget, the stall-free batching of Sarathi-Serve and the default mode of
vLLM, gets both: the budget bounds the length of every step, hence the time between
tokens, and the chunks fill the capacity the decodes leave free. The planner is a few
lines, and its mistakes are exactly the latency pathologies it exists to prevent.
