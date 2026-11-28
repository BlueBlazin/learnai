Implement the scheduling loop of a continuous-batching LLM server. Before every forward pass, decide which waiting requests enter the batch and which running sequences continue decoding, so new requests join on the next step and finished ones free their slots immediately.

![This problem is the Engine loop, which picks the sequences of every forward pass.](assets/map.svg)

## Specification

```python
async def run(engine, queue: asyncio.Queue, max_running: int) -> None: ...
```

`run` is a coroutine that runs indefinitely. It drives a [`SimEngine`](../../../python/src/learnai/kit/simengine.py), the kit's simulated inference engine that models forward-pass timing without running a real model. Time is virtual and measured in **milliseconds**.

**The engine.** You interact with `engine` through three operations:

- `engine.admit(request)` turns a request into a *live sequence* and takes no virtual time. At most `max_running` sequences may be live at once (the engine has `max_running` *slots*).
- `result = await engine.step(prefill=[(id, n_tokens), ...], decode=[id, ...])` runs one forward pass and returns when it finishes. Each sequence in `prefill` is passed with its full prompt (`n_tokens = request.prompt_tokens`) and produces its **first** output token at the end of that step. Each sequence in `decode` produces one more token. A sequence cannot appear in both lists during the same step, because it begins decoding on the *next* step after its prefill.
- `result.finished` contains the ids of the sequences that produced their last token during that step. They stop being live and free their slots automatically, with no extra call needed.

A step that prefills $P$ total prompt tokens and decodes $D$ sequences takes $8 + 0.25\,P + D$ milliseconds. Only one step can run at a time. Violating any engine rule (exceeding `max_running` live sequences, running an empty step, placing an id in both lists, decoding an unknown or finished id, or starting a second step while one is running) raises `ProtocolViolation` and fails the test even if you catch it.

**The requests.** `queue` is an `asyncio.Queue` of [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. The test pushes each request into `queue` at its arrival time. You may read `request.id` (opaque, hashable, unique) and `request.prompt_tokens`. Do not read `request.output_tokens`, because a real server does not know how long a generation will be until the sequence finishes.

**The policy.** A request that has been taken from `queue` but not yet admitted is *waiting*. An admitted sequence that has not finished is *running*. Before every step:

1. **Collect.** Move every request currently in `queue` to the back of the waiting list, preserving queue order.
2. **Admit.** While fewer than `max_running` sequences are running and at least one request is waiting, admit the oldest waiting request (first come, first served).
3. **Step.** Call `engine.step` once. Pass the sequences admitted in step (2) as `prefill` (in admission order, each with its full prompt) and every sequence that was already running before step (2) as `decode` (in admission order).

When the step returns, remove every id in `result.finished` from the running set and immediately plan the next step at the same virtual instant.

**Idle behavior.** If nothing is waiting and nothing is running after step (1), there is no work to run. Block on `await queue.get()` until a request arrives, then continue with step (1). Never call `engine.step` with two empty lists, and never sleep or poll.

**Events at the same instant.**

- Requests that arrive at the same virtual instant are pushed into `queue` in arrival order, and all of them are present before your coroutine resumes from `queue.get()`.
- Any request arriving *during* a step is already in `queue` when the step returns, so step (1) collects it before the next step. This includes requests arriving at the exact instant a step finishes, because arrivals are delivered before step completion.
- As a result, no free slot is ever left empty while a request is waiting, and the engine is never idle while a request is waiting or a sequence is running.

## Example

Let `max_running = 2` with six requests given as `(arrival ms, prompt tokens, output tokens)`: request 0 `(0, 8, 4)`, request 1 `(0, 16, 2)`, request 2 `(5, 8, 3)`, request 3 `(45, 4, 2)`, request 4 `(50, 12, 1)`, and request 5 `(60, 4, 2)`.

| step | starts | prefill | decode | takes | at its end |
|---|---|---|---|---|---|
| 0 | 0 | `(0, 8), (1, 16)` | | 8 + 6 + 0 = 14 | first tokens of 0 and 1 |
| 1 | 14 | | `0, 1` | 8 + 0 + 2 = 10 | request 1 finishes (2 tokens) |
| 2 | 24 | `(2, 8)` | `0` | 8 + 2 + 1 = 11 | first token of 2 |
| 3 | 35 | | `0, 2` | 10 | request 0 finishes (4 tokens) |
| 4 | 45 | `(3, 4)` | `2` | 8 + 1 + 1 = 10 | request 2 finishes |
| 5 | 55 | `(4, 12)` | `3` | 8 + 3 + 1 = 12 | requests 4 and 3 finish |
| 6 | 67 | `(5, 4)` | | 9 | first token of 5 |
| 7 | 76 | | `5` | 9 | request 5 finishes |

![The worked example, one row per request. Request 2 takes the slot of request 1 at 24 ms while request 0 keeps decoding.](assets/timeline.svg)

Request 2 arrives at 5 ms during step 0, but both slots are occupied until request 1 finishes at 24 ms. At that point, request 2 joins step 2 alongside the decode of request 0, demonstrating continuous batching. Request 3 arrives at 45 ms, the exact instant step 3 frees request 0's slot, and enters step 4 immediately. Request 4 has only one output token, so it finishes at the end of its prefill step. Request 5 arrives at 60 ms while both slots are full, and then runs alone at 67 ms once both slots free up.

The end-to-end latencies (from arrival to last token) are 45, 24, 50, 22, 17, and 25 ms.

## Constraints

- `1 <= max_running <= 16`. Prompts have 1 to 128 tokens and outputs have 1 to 40 tokens. Arrival times are multiples of 0.25 ms, so all timestamps in tests are exact floats.
- The policy uniquely determines every step. Tests compare your sequence of steps (start time, `prefill`, `decode`) exactly against literal schedules on small scenarios and against the reference schedule on seeded bursty and Poisson workloads of up to 120 requests. Tests also verify the scheduling invariants: every request finishes, admission is first come, first served, no slot stays free while a request waits, every running sequence decodes on every step, and the engine never sits idle when work exists.
- On a Poisson workload of 150 requests at 12 requests per second with `max_running = 8`, both the median end-to-end latency and the 99th-percentile time to first token must be at most **half** of what static batching (admitting a new batch only after the previous batch finishes completely) achieves on the same workload.
- Use only `asyncio` and the engine. Because time is virtual, the real clock and real sleeping have no meaning here.

## Why this matters

Autoregressive generation is a loop of forward passes, and requests require very different numbers of passes. One answer might be 5 tokens long while the next is 500. A static-batching server that runs each batch to completion forces short requests to wait behind long ones, wastes slots on padding, and queues new arrivals behind the entire batch.

Scheduling per iteration instead of per request (introduced by Orca and adopted by vLLM, TGI, SGLang, TensorRT-LLM, and llama.cpp's server) is the largest throughput and latency improvement in LLM serving. Refilling slots the moment they free up keeps the GPU busy and decouples time to first token from other requests' output lengths. Every other scheduling feature in this track (priorities, deadlines, chunked prefill, and fairness) modifies step 2 or step 3 of this loop.
