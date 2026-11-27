An agent drives a chat model in a loop by calling the model, running any tools it requests, appending the tool outputs to the conversation, and calling the model again until it produces a plain-text reply. Implement this loop while respecting the chat-completions protocol, running parallel tool calls concurrently, and handling tool or argument errors without crashing or looping forever.

## Specification

```python
async def run_agent(model, tools: dict, messages: list[dict], max_steps: int) -> list[dict]: ...
```

**Messages and transcript.** `messages` is a list of chat-completions message dicts representing the conversation so far (system, user, and possibly prior assistant and tool messages). Return a new list, `transcript`, starting with all elements of `messages` in order, followed by every message appended during the loop. Do not mutate `messages`.

**Model.** Calling `await model.acomplete(transcript)` performs one model call and returns an assistant message dict in one of two shapes:

```python
{"role": "assistant", "content": "It is 21 C in Paris."}  # final answer
{
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        },
    ],
}
```

The `"tool_calls"` key appears only when the model requests tools, and when present it contains one or more call dicts. Each call's `"arguments"` value is a string that is expected to be a JSON object, though malformed or non-object JSON can occur. Do not assume any naming pattern for `"id"`. On every call, the model validates the conversation history and raises `ChatProtocolError` if an assistant message with `"tool_calls"` is not immediately followed by exactly one tool message for each requested call ID.

**Tools.** `tools` maps tool names (strings) to kit `ScriptedTool` objects. Calling `await tool(**arguments)` runs the tool asynchronously and either returns a result (a string or any JSON-serializable value) or raises an exception.

**Loop.** One *step* is one model call. Repeat at most `max_steps` times:

1. Call `await model.acomplete(transcript)` and append the returned assistant message unchanged.
2. If the message has no `"tool_calls"` (or it is empty), return `transcript`.
3. Otherwise, execute the requested tool calls to produce one string **outcome** per call (using the rules below), and append one tool message per call in the exact order of `"tool_calls"`:

   ```python
   {"role": "tool", "tool_call_id": call["id"], "content": outcome}
   ```

If the model still requests tools on step `max_steps`, execute those calls and append their tool messages before returning `transcript` without making another model call. This guarantees that the returned transcript is always a valid conversation that a later `run_agent` call can continue.

**Tool outcomes.** For each call, let `function = call["function"]` and `name = function["name"]`. Determine the outcome string using the first matching rule:

| Situation | Tool runs? | Outcome |
|---|---|---|
| `function["name"]` is not in `tools` | No | `json.dumps({"error": "unknown tool: " + name})` |
| `function["arguments"]` is invalid JSON, or valid JSON that is not a JSON object (`dict`) | No | `json.dumps({"error": "invalid arguments"})` |
| `await tool(**arguments)` raises an `Exception` `exc` | Yes | `json.dumps({"error": "tool failed: " + str(exc)})` |
| `await tool(**arguments)` returns `result` | Yes | `result` if `isinstance(result, str)`, else `json.dumps(result)` |

Call `json.dumps` with its default arguments. Errors are returned to the model as normal tool outcomes so the model can recover on the next step. Never let a bad tool call or a failing tool raise out of `run_agent`.

**Concurrency.** All tool calls within a single step are independent and must run concurrently. Every call in the step that invokes a tool must start as soon as the model's message arrives, in the order listed in `"tool_calls"`. Wait until all calls in the step have finished or failed. A failure in one tool must neither cancel nor delay its siblings. If the model requests the same tool twice in one step, run it twice. After all calls complete, append their tool messages in call order (regardless of completion order) before calling the model again.

Use `asyncio` (`asyncio.gather` or tasks) rather than wall-clock timers, because tests run on a simulated event-loop clock.

## Example

Suppose `get_weather` takes `0.5` s, `get_time` takes `0.25` s, and each model call takes `1.0` s. Consider calling `run_agent(model, tools, [{"role": "user", "content": "Weather and time in Paris?"}], 8)` when the model finishes in three steps:

| Time (s) | Event |
|---|---|
| 0.0 | Step 1: call the model |
| 1.0 | Model requests `get_weather({"city": "Paris"})` as `call_0` and `get_time({"zone": "CET"})` as `call_1`. Both tools start now. |
| 1.25 | `get_time` returns `{"zone": "CET", "time": "14:05"}` |
| 1.5 | `get_weather` returns `"21 C in Paris"`. Step 2: call the model. |
| 2.5 | Model requests `get_weather` with arguments `'{"city": '` as `call_2` and `get_stock({"symbol": "X"})` as `call_3`. Neither tool runs. Step 3: call the model. |
| 3.5 | Model answers `"It is 21 C in Paris at 14:05."` |

The returned transcript contains the initial user message followed by these seven messages (eight in total):

```python
{"role": "assistant", "content": None, "tool_calls": [...]}  # call_0, call_1
{"role": "tool", "tool_call_id": "call_0", "content": "21 C in Paris"}
{"role": "tool", "tool_call_id": "call_1", "content": '{"zone": "CET", "time": "14:05"}'}
{"role": "assistant", "content": None, "tool_calls": [...]}  # call_2, call_3
{"role": "tool", "tool_call_id": "call_2", "content": '{"error": "invalid arguments"}'}
{"role": "tool", "tool_call_id": "call_3", "content": '{"error": "unknown tool: get_stock"}'}
{"role": "assistant", "content": "It is 21 C in Paris at 14:05."}
```

Even though `get_time` finishes first, its tool message appears second to match the order of `"tool_calls"`. The string result `"21 C in Paris"` is kept as-is, while the dict result is JSON-encoded. Running the two tools sequentially would take `0.75` s instead of `0.5` s in step 1, bringing the total time to `3.75` s.

With `max_steps=1`, the loop finishes at `1.5` s and returns the first four messages: the user message, the first assistant message, and both tool responses.

## Constraints

- `max_steps >= 1`, with at most 16 tool calls per step and at most 40 steps. Tool return values are JSON-serializable.
- The returned transcript is compared exactly, message by message, and the model must not report any `ChatProtocolError`.
- **Call accounting.** The number of model calls must equal the exact number of steps required (never exceeding `max_steps`). Tools must run once per call that reaches the last two rows of the outcome table, with the exact arguments supplied by the model, and never for unknown names or invalid arguments.
- **Simulated time.** All test durations are exact multiples of 1/64 s. The simulated time when `run_agent` returns must equal the sum over steps of the model latency plus the **longest** tool latency in that step. Every tool in a step must start at the exact instant the model's message arrives.
- Exceeding `max_steps` causes the scripted model to raise, and each test case has a 5-second real-time limit.

## Why this matters

Every tool-using agent framework is built around this loop, and most production agent failures trace back to one of its edge cases. Letting a tool exception escape turns a recoverable error into a 500 response instead of letting the model inspect the error and retry. Attaching a tool result to the wrong `tool_call_id` passes API validation silently while feeding the model mismatched data. Dropping one response from a batch of parallel tool calls breaks the conversation protocol on the next request. Awaiting tools sequentially multiplies latency across parallel calls, and omitting a step limit turns a stuck model into an unbounded API bill.
