Implement `run_agent`, an asynchronous tool-calling loop that starts read-only tools before the model's turn finishes. It launches a read-only tool as soon as a predictor guesses its arguments or its streamed arguments finish. It keeps a speculative result only on an exact match, cancels mismatches, and waits until the turn ends before starting any tool with side effects.

## Specification

```python
async def run_agent(
    model, tools: dict, messages: list[dict], predictor, max_steps: int
) -> list: ...
```

The transcript format, tool outcomes, message ordering, and step limit follow `tool-call-loop` and are specified below. Unlike that problem, the model streams each turn incrementally, and read-only tools start before the turn finishes.

**The stream.** Calling `model.stream(transcript)` starts one model turn and returns an async iterator over its *pieces* (`async for piece in model.stream(transcript)`). The pieces of a turn arrive in this order:

```python
{"content": "Let me "}  # zero or more pieces of text
{
    "tool_calls": [
        {
            "index": 0,
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        }
    ]
}  # the header of call 0: its id and name
{"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}  # an argument fragment
{"tool_calls": [{"index": 0, "function": {"arguments": ' "Paris"}'}}]}
# ... the header of call 1, its fragments, and so on ...
{"finish_reason": "tool_calls"}  # always the last piece; "stop" if there are no calls
```

Calls are streamed sequentially. Once the header of call $i + 1$ arrives, no more fragments of call $i$ follow. Calling `assemble(pieces)` from `learnai.kit` (`from learnai.kit import assemble`) combines all pieces of a turn into the assistant message, which is appended to the transcript unchanged. When `model.stream` is called, the model validates the conversation history as in `tool-call-loop`.

**The tools.** `tools` maps tool names to kit `ScriptedTool` instances. Calling `await tool(**arguments)` runs the tool asynchronously. `tool.side_effects` is `False` for a read-only tool (which may be started speculatively and abandoned) and `True` for a tool that modifies state. Canceling the task awaiting a tool stops its execution.

**The predictor.** `predictor(name, text)` is a plain synchronous function. Given a tool's name and the argument text received so far (`""` when only the header has arrived), it returns a `dict` of expected arguments, or `None` if it has no prediction.

**When a call's tool is started.** Three instants matter for call $i$:

- Its **header** arrives.
- It is **complete** when the first piece that does not belong to it arrives (either the header of call $i + 1$ or the finish piece).
- The **turn ends** when the finish piece arrives.

Let `name` be the call's tool name.

1. **Speculation.** Suppose `name` is a key in `tools` and `tools[name].side_effects` is `False`. When the header arrives, and again after each argument fragment, call `predictor(name, text)` with the argument text received so far, as long as no run has started for this call. If the predictor returns a dict `prediction`, start `tool(**prediction)` at that instant. That task is the call's *speculative run*, and you do not call the predictor again for this call. Never start a speculative run for an unknown tool name or a tool with side effects.
2. **Completion.** Let `final` be the complete argument text parsed as JSON. At completion, apply the first matching row:

   | situation | at completion |
   |---|---|
   | `name` is not a key in `tools` | record outcome `{"error": "unknown tool: <name>"}` and run nothing |
   | the text is not valid JSON, or not a JSON object | discard any speculative run and record outcome `{"error": "invalid arguments"}` |
   | there is a speculative run and `prediction == final` | keep the speculative run as the call's run |
   | the tool has no side effects | discard any speculative run and start `tool(**final)` at this instant |
   | the tool has side effects | start `tool(**final)` when the turn ends, never earlier |

   `prediction == final` is standard Python dict equality. Whitespace and key order in the model's JSON text do not matter, while any missing, extra, or different value causes a mismatch.
3. **Discarding.** To discard a speculative run, cancel its task if it is still running and ignore its result. A discarded run that has already finished or raised an exception has no further effect.
4. **Outcomes.** When a tool runs, its outcome follows `tool-call-loop`. Use the return value `result` directly if it is a `str`, `json.dumps(result)` otherwise, or `json.dumps({"error": "tool failed: " + str(exc)})` if it raises an exception `exc`. Error outcomes from the table above are serialized as `json.dumps({"error": ...})` using the exact messages shown.

**The loop.** A step is one model turn. Repeat up to `max_steps` times. In each step, stream a turn and append its assembled assistant message to the transcript. If the turn has no tool calls, return the transcript. Otherwise, wait until the turn has ended and every kept or started tool run has finished. Append one message `{"role": "tool", "tool_call_id": call["id"], "content": outcome}` per call in call order, then start the next turn immediately. After step `max_steps`, return the transcript with that turn's tool calls answered. Do not modify `messages`. When `run_agent` returns, no tool run may still be in progress.

Use `asyncio` primitives only (tasks, `ensure_future`, `cancel`), never the real clock.

## Example

One piece arrives every 0.125 s, carrying 8 characters of text. `get_weather` takes 0.6875 s and `search` takes 1.0625 s (both read-only). `send_email` takes 0.4375 s and has side effects. The predictor assumes the JSON object ends immediately after the text seen so far, returning `json.loads(text + "}")` when that produces a dict and `None` otherwise.

Turn 1 requests `get_weather({"city": "Paris"})` and `send_email({"to": "ann"})`. Turn 2 requests `search` with the text `{"q":"x","limit":1}`. Turn 3 outputs `Done.`.

| time | piece | what happens |
|---|---|---|
| 0.125 | header of `call_0`, `get_weather` | predictor on `""` returns `None` |
| 0.25 | `{"city":` | predictor returns `None` |
| 0.375 | ` "Paris"` | predictor returns `{"city": "Paris"}`. **`get_weather` starts** |
| 0.5 | `}` | already speculating |
| 0.625 | header of `call_1`, `send_email` | `call_0` is complete and matches the prediction, so it is kept. `send_email` has side effects, so no prediction runs |
| 0.75, 0.875 | `{"to": "`, `ann"}` | |
| 1.0 | finish | `call_1` is complete and the turn ends. **`send_email` starts** |
| 1.0625 | | `get_weather` returns `"21 C in Paris"` |
| 1.4375 | | `send_email` returns, both tool messages are appended, and turn 2 begins |
| 1.5625 | header of `call_2`, `search` | predictor returns `None` |
| 1.6875 | `{"q":"x"` | predictor returns `{"q": "x"}`. **`search(q="x")` starts** |
| 1.8125, 1.9375 | `,"limit"`, `:1}` | |
| 2.0625 | finish | final is `{"q": "x", "limit": 1}`, which mismatches the prediction. The run is **canceled** and **`search(q="x", limit=1)` starts** |
| 3.125 | | `search` returns `["x result 0"]` and turn 3 begins |
| 3.375 | | `Done.` arrives (one text piece and the finish piece), and `run_agent` returns |

The run finishes in 3.375 s. The non-speculative loop of `tool-call-loop` takes 3.625 s on the same stream because it waits until 1.0 s to start `get_weather`, finishing that tool at 1.6875 s. The wrong guess in turn 2 costs one canceled run and adds no latency. If `search(q="x")` had been reused for the real call, the model would have received two results when it asked for one.

## Constraints

- `max_steps` $\ge 1$, with at most 8 tool calls per turn and at most 8 turns. The predictor is deterministic and does not raise.
- The returned transcript is compared exactly, and the model must not report a protocol violation.
- **Tool executions.** For every tool, the set of runs is compared exactly against the rules above, checking arguments, start time, completion status (done, failed, or canceled), and end time. An extra run, a late start, or a speculative run not canceled at call completion fails the test. A tool with side effects must only start at the end of a turn.
- **Time.** The simulated return time of `run_agent` must not exceed the time prescribed by the rules. Test output reports the non-speculative loop's time alongside it.
- All event times are multiples of 1/16 s, so floating-point arithmetic is exact. Times are chosen so no tool finishes at the same instant a piece arrives, meaning simultaneous event ordering never affects the result.
- Each test case times out after 5 seconds of real time.

## Why this matters

In a standard agent loop, model generation and tool execution run sequentially, so the user waits for the sum of both latencies. Tool calls are slow to stream yet often easy to predict. A model may take a second to spell out a search query whose first words already determine it, and agents frequently repeat patterns such as running tests after every edit. Starting read-only tools while the model is still streaming hides tool latency behind token generation, saving a large fraction of total runtime across multi-step tasks.

The risks scale directly with that speedup. Reusing a speculative result when the model's final arguments differ slightly produces a wrong answer with no obvious error in the logs. Failing to cancel discarded runs leaks database connections or browser sessions. Worse, launching a side-effecting tool like `send_email` or `delete_file` speculatively cannot be undone, making the `side_effects` flag a hard safety boundary rather than an optimization hint.
