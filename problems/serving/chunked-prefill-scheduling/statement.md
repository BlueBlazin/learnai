Implement the step planner for a continuous-batching engine operating under a per-step token budget. Each step processes at most `max_batch_tokens` tokens, decoding running sequences first and splitting prompts into chunks with the remaining budget so long prompts never stall active generations.

![This problem is the Engine loop, here the plan of one step under a token budget.](assets/map.svg)

## Specification

```python
def plan(running, prefilling, waiting, max_batch_tokens, long_prefill_threshold):
    ...
    return prefill, decode
```

`plan` is a pure function called once before each engine step with the current server state. It returns the tuple `(prefill, decode)` passed directly to `engine.step`.

| Argument | Meaning |
|---|---|
| `running` | Ids of sequences whose prompts are fully prefilled, in admission order. Each sequence can decode 1 token in this step. |
| `prefilling` | List of `(id, remaining_tokens)` pairs for admitted sequences whose prompts are only partly prefilled, in admission order, with `remaining_tokens >= 1`. |
| `waiting` | List of `(id, prompt_tokens)` pairs for arrived requests that are not yet admitted, in arrival order. |
| `max_batch_tokens` | Maximum total tokens allowed in one step (`>= 1`), counting both prompt tokens and decoded sequences. |
| `long_prefill_threshold` | Maximum prompt tokens a single sequence may receive in one step (`>= 1`). |

The return value is `(prefill, decode)`, where `prefill` is a list of `(id, n_tokens)` chunks and `decode` is a list of sequence ids. Build these two lists in order:

1. **Decodes first.** Copy `running` in order into `decode`. Each decoded sequence uses 1 token of the step budget. If `running` contains more sequences than `max_batch_tokens`, take only the first `max_batch_tokens` ids.
2. **Prompts next, first come, first served.** Iterate through `prefilling` and then `waiting` in order. Assign each candidate a chunk of size

   $$
   n = \min(\text{remaining tokens},\ \text{long\_prefill\_threshold},\ \text{budget left}),
   $$

   and subtract $n$ from the remaining budget. Stop as soon as the remaining budget reaches 0. Candidates that are not reached receive nothing in this step and do not appear in `prefill`.

A prompt that does not fit within the remaining budget or `long_prefill_threshold` is **split** across steps. It takes as many tokens as allowed now and continues in later steps. No candidate is skipped while budget remains, because the next candidate in line always receives at least 1 token.

Each sequence appears at most once across `prefill` and `decode`. In particular, a sequence that receives its final prompt chunk in this step is *not* included in `decode` during the same step. That final chunk produces the sequence's first output token, and the sequence moves into `running` on the next call.

**Engine loop.** The test suite drives `plan` using the loop below and a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) configured with `max_batch_tokens`. You do not implement this loop.

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

Time is virtual and measured in **milliseconds**. A step that processes $P$ prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms. A sequence emits its first output token at the end of the step that finishes its prompt, and emits one output token per subsequent step in which it is decoded.

There is no limit on the number of live sequences beyond the token budget. The engine rejects a step and fails the test if the step exceeds the budget, is empty, lists any sequence twice, decodes a sequence whose prompt is incomplete, or gives a sequence more prompt tokens than it has remaining.

A request that arrives during a step, or at the exact instant a step ends, appears in `waiting` on the next call.

## Example

Let `max_batch_tokens = 32` and `long_prefill_threshold = 24`. Three requests arrive, written as `(arrival, prompt_tokens, output_tokens)`: request 0 is `(0, 16, 4)`, request 1 is `(0, 40, 2)`, and request 2 is `(3, 8, 2)`.

| starts | `running` | `prefilling` | `waiting` | `prefill` | `decode` | takes |
|---|---|---|---|---|---|---|
| 0 | | | `(0, 16), (1, 40)` | `(0, 16), (1, 16)` | | 8 + 8 = 16 |
| 16 | `0` | `(1, 24)` | `(2, 8)` | `(1, 24), (2, 7)` | `0` | 8 + 7.75 + 1 = 16.75 |
| 32.75 | `0, 1` | `(2, 1)` | | `(2, 1)` | `0, 1` | 8 + 0.25 + 2 = 10.25 |
| 43 | `0, 2` | | | | `0, 2` | 10 |

![The four steps of the example, each against the budget of 32 tokens. Decodes come first, and request 1's 40-token prompt enters as chunks of 16 and 24.](assets/step-budget.svg)

At time 0, the budget is 32. Request 0 takes its full 16-token prompt, leaving 16 tokens for request 1.

At time 16, decoding request 0 costs 1 token, leaving 31. Request 1 is capped at `long_prefill_threshold = 24`, which finishes its prompt. Request 2, which arrived at time 3, takes the remaining 7 tokens.

At time 32.75, request 1 has moved to `running` and decodes alongside request 0, while request 2 takes its last prompt token. The times to first token are 16, 32.75, and 40 ms.

## Constraints

- `1 <= max_batch_tokens <= 512` and `1 <= long_prefill_threshold <= 512`. Prompts have between 1 and 4,000 tokens. Sequence ids are opaque. Do not mutate the input lists.
- `plan` is deterministic, and the tests compare its output exactly. Coverage includes direct calls with literal expected outputs, 300 random states compared against the reference, and seeded Poisson workloads of 60 requests run through the engine loop above (checked against literal steps on small scenarios and against the reference on large ones).
- **Stall bound.** Six sequences are generating when a 4,000-token prompt arrives, followed by three 24-token prompts. With `max_batch_tokens = 512` and `long_prefill_threshold = 512`, the longest gap between two consecutive tokens of any of the six running sequences must be at most **150 ms**. The largest possible step takes $8 + 0.25 \cdot 506 + 6 = 140.5$ ms, whereas prefilling the prompt in one step would take over a second.
- **Head-of-line bound.** In the same scenario with `long_prefill_threshold = 128`, each of the three short prompts must emit its first token within **120 ms** of arrival, and the 150 ms inter-token gap bound must still hold.

## Why this matters

Prefill and decode have opposite compute profiles. A long prefill pushes thousands of tokens through the model in one pass and takes hundreds of milliseconds, while a decode step processes one token per sequence and takes only a few milliseconds. Mixing an unchunked prompt into a decode batch freezes every active stream for the entire prefill, so users watch text stop for a second whenever someone else submits a long document.

Running prefills separately protects streaming latency but leaves GPU compute underutilized. Chunked prefill with a token budget, popularized as stall-free batching in Sarathi-Serve and now the default in vLLM, achieves both goals. The token budget caps step duration and inter-token latency, while prompt chunks fill the compute capacity left free by decodes.
