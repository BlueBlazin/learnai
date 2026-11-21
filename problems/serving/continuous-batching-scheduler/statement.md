Implement the scheduling loop of a continuous-batching LLM server: before every
forward pass it decides which sequences take part, so that a new request joins the
running batch at the next step and a finished one leaves at once. The difficulty is in
the bookkeeping around a step: who may be decoded, when a slot is free, and what the
loop does when there is nothing to do.

![This problem is the Engine loop, which picks the sequences of every forward pass.](assets/map.svg)

## Specification

```python
async def run(engine, queue: asyncio.Queue, max_running: int) -> None: ...
```

`run` is a coroutine that never returns. It drives a
[`SimEngine`](../../../python/src/learnai/kit/simengine.py), the kit's simulated
inference engine: a forward pass with the model taken out. Time is virtual and this
problem counts it in **milliseconds**.

**The engine.** You need three things from it:

- `engine.admit(request)` makes a request a *live sequence*. It takes no time. At most
  `max_running` sequences may be live at once (the engine has `max_running` *slots*).
- `result = await engine.step(prefill=[(id, n_tokens), ...], decode=[id, ...])` runs
  one forward pass and returns when it ends. A sequence listed in `prefill` with its
  whole prompt (`n_tokens = request.prompt_tokens`) produces its **first** output token
  at the end of that step. A sequence listed in `decode` produces one more token. A
  sequence cannot be in both lists of one step: after its prefill step it is decoded
  from the *next* step on.
- `result.finished` holds the ids of the sequences that produced their last token in
  that step. They are no longer live, and their slots are free, without a call.

A step over $P$ prompt tokens in total and $D$ decoded sequences takes
$8 + 0.25\,P + D$ milliseconds. One step runs at a time. Anything the engine cannot do
(more than `max_running` live sequences, an empty step, an id in both lists, decoding
an unknown or finished id, a second step while one runs) raises a `ProtocolViolation`
and fails the case, also if you catch the exception.

**The requests.** `queue` is an `asyncio.Queue` of
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. The test puts
each request into it at the request's arrival time. You may read `request.id` (opaque,
hashable, unique) and `request.prompt_tokens`. Do not read `request.output_tokens`: a
server learns the length of an output when the sequence finishes.

**The policy.** A request that has left the queue and has not been admitted is
*waiting*; an admitted sequence that has not finished is *running*. Before every step:

1. **Collect.** Move every request that is in `queue` to the end of the waiting list,
   in queue order.
2. **Admit.** While fewer than `max_running` sequences are running and a request is
   waiting, admit the oldest waiting request (first come, first served).
3. **Step.** Call `engine.step` once: `prefill` lists the sequences admitted in (2),
   in admission order, each with its whole prompt; `decode` lists every sequence that
   was running before (2), in admission order.

When the step returns, the sequences in `result.finished` stop running, and the loop
plans the next step immediately, at the same virtual instant.

**Idle.** If nothing is waiting and nothing is running after (1), there is no step to
run. Block on `await queue.get()` until a request arrives, then continue with (1).
Never call `engine.step` with two empty lists, and never sleep or poll.

**Events at one instant.**

- Requests that arrive at the same instant are put into the queue in arrival order,
  and all of them are in the queue before your coroutine resumes from `queue.get()`.
- A request that arrives *during* a step is in the queue when the step returns, so it
  is collected before the next step. This includes a request that arrives at exactly
  the instant the step ends: arrivals are delivered before the step's completion.
- Consequently no free slot is ever left empty while a request waits, and the engine
  is never idle while a request waits or a sequence runs.

## Example

`max_running = 2`. Six requests, written as (arrival ms, prompt tokens, output tokens):
request 0 `(0, 8, 4)`, request 1 `(0, 16, 2)`, request 2 `(5, 8, 3)`, request 3
`(45, 4, 2)`, request 4 `(50, 12, 1)`, request 5 `(60, 4, 2)`.

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

Request 2 arrives at 5, during step 0, but both slots are taken until request 1
finishes at 24; it joins step 2 next to the decode of request 0, which is the point of
continuous batching (Figure 2). Request 3 arrives at exactly 45, the instant step 3
ends and frees the slot of request 0, and is admitted into step 4. Request 4 has a
single output token, so it finishes at the end of its prefill step. Request 5 arrives
at 60, during step 5, when both slots are taken; at 67 both are free and it runs alone.

The end-to-end latencies (arrival to last token) are 45, 24, 50, 22, 17, and 25 ms.

## Constraints

- `1 <= max_running <= 16`. Prompts have 1 to 128 tokens and outputs 1 to 40. Arrival
  times are multiples of 0.25 ms, so every time in a test is an exact float.
- The policy determines every step. The tests compare the sequence of steps (start
  time, `prefill`, `decode`) exactly: with literal schedules on small scenarios, and on
  seeded bursty and Poisson workloads of up to 120 requests with the reference's steps
  and with the invariants above (every request finishes; first come, first served; no
  free slot while a request waits; every running sequence decodes in every step; no
  idle engine while there is work).
- On a Poisson workload of 150 requests at 12 requests per second with
  `max_running = 8`, the median end-to-end latency and the 99th-percentile time to
  first token must each be at most **half** of what static batching (admit a new batch
  only when the previous batch has finished entirely) achieves on the same workload.
- Use `asyncio` and the engine only. Time is virtual: the real clock and real sleeping
  have no meaning here.

## Why this matters

Generation is a loop of forward passes, and requests need very different numbers of
them: one answer is 5 tokens long, the next 500. A server that forms a batch and runs
it to completion makes the short request wait for the long one and leaves its slot
computing padding, while new arrivals queue behind the whole batch. Scheduling per
iteration instead of per request, introduced by Orca and used by vLLM, TGI, SGLang,
TensorRT-LLM, and llama.cpp's server, is the single largest throughput and latency
improvement in LLM serving: slots are refilled the moment they free up, so the GPU
stays full and time to first token stops depending on other people's output lengths.
Every other scheduling feature in this track (priorities, deadlines, chunked prefill,
fairness) is a change to step 2 or step 3 of this loop.
