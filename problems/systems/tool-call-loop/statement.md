An agent is a loop around a chat model: call the model, run the tools it asks for,
append their results, call it again, until it answers in plain text. Write that
loop. The model's side of the protocol is strict (every tool call must be answered,
by id, before the model is called again), tools fail, models emit calls to tools
that do not exist and arguments that are not JSON, and none of that may crash the
loop or let it run forever.

## Specification

```python
async def run_agent(model, tools: dict, messages: list[dict], max_steps: int) -> list[dict]: ...
```

**Messages** are dicts in the chat-completions format. `messages` is the
conversation so far (system, user, and possibly earlier assistant and tool
messages). The function returns the **transcript**: a new list that starts with
the messages of `messages`, in order, followed by everything the loop appends.
`messages` itself is not modified.

**The model.** `await model.acomplete(transcript)` makes one model call and returns
the assistant message, one of

```python
{"role": "assistant", "content": "It is 21 C in Paris."}  # a final answer
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

The key `tool_calls` is present only when the model requests tools, and then holds
one or more calls. `arguments` is a **string** that should be a JSON object and is
not always one. Ids are whatever the model chose; do not assume a pattern. The model
validates every conversation it is given and raises `ChatProtocolError` if an
assistant message with tool calls is not followed directly by exactly one tool
message for each of its ids.

**The tools.** `tools` maps a tool's name to a kit `ScriptedTool`.
`await tool(**arguments)` runs it, takes some time, and returns its result (a
string or any JSON-serializable value) or raises an exception.

**The loop.** A *step* is one model call. Repeat at most `max_steps` times:

1. Call the model with the transcript and append its message, unchanged.
2. If the message has no tool calls, it is the final answer: return the transcript.
3. Otherwise produce one **outcome** per tool call (below), then append one tool
   message per call, **in the order of the calls**:

   ```python
   {"role": "tool", "tool_call_id": call["id"], "content": outcome}
   ```

If the model still requested tools at step `max_steps`, those calls are executed
and answered like any others, and then the transcript is returned without another
model call. The transcript is therefore always a valid conversation that a later
call of `run_agent` can continue.

**The outcome of a call** is a string, decided by the first rule that applies:

| situation | the tool runs? | outcome |
|---|---|---|
| `function.name` is not a key of `tools` | no | `json.dumps({"error": "unknown tool: " + name})` |
| `function.arguments` is not valid JSON, or is JSON but not an object | no | `json.dumps({"error": "invalid arguments"})` |
| `await tool(**arguments)` raises an `Exception` `exc` | yes | `json.dumps({"error": "tool failed: " + str(exc)})` |
| it returns `result` | yes | `result` if it is a `str`, else `json.dumps(result)` |

`json.dumps` is called with its default arguments. An error is an outcome like any
other: it goes to the model, which can correct itself in the next step. The loop
itself never raises because of a bad call or a failing tool.

**Concurrency.** The calls of one step are independent, so they run at the same
time. Every call of the step that runs a tool is started at the instant the model's
message arrives, in the order of the calls; the step waits until all of them have
finished or failed; a failure neither cancels nor delays the others. A tool that is
requested twice in one step runs twice. Only then are the tool messages appended
(in call order, whatever order the tools finished in) and the model called again.

Time is the event loop's: use `asyncio` (`gather`, tasks), never the real clock.

## Example

`get_weather` takes 0.5 s and `get_time` 0.25 s; a model call takes 1 s.
`run_agent(model, tools, [{"role": "user", "content": "Weather and time in Paris?"}], 8)`
with a model that answers in three steps:

| time | event |
|---|---|
| 0.0 | step 1: the model is called |
| 1.0 | it requests `get_weather({"city": "Paris"})` as `call_0` and `get_time({"zone": "CET"})` as `call_1`; both start now |
| 1.25 | `get_time` returns `{"zone": "CET", "time": "14:05"}` |
| 1.5 | `get_weather` returns `"21 C in Paris"`; step 2: the model is called |
| 2.5 | it requests `get_weather` with arguments `{"city": ` as `call_2` and `get_stock({"symbol": "X"})` as `call_3`; neither runs; step 3: the model is called |
| 3.5 | it answers `"It is 21 C in Paris at 14:05."` |

The transcript has eight messages: the user message, then

```python
({"role": "assistant", "content": None, "tool_calls": [...]},)  # call_0, call_1
({"role": "tool", "tool_call_id": "call_0", "content": "21 C in Paris"},)
({"role": "tool", "tool_call_id": "call_1", "content": '{"zone": "CET", "time": "14:05"}'},)
({"role": "assistant", "content": None, "tool_calls": [...]},)  # call_2, call_3
({"role": "tool", "tool_call_id": "call_2", "content": '{"error": "invalid arguments"}'},)
({"role": "tool", "tool_call_id": "call_3", "content": '{"error": "unknown tool: get_stock"}'},)
({"role": "assistant", "content": "It is 21 C in Paris at 14:05."},)
```

`get_time` finished first, and its result is still the second tool message: the
order is the order of the calls. The string result is passed through as it is; the
dict is encoded. Run one after the other, the two tools of step 1 would have taken
0.75 s instead of 0.5 s, and the whole run 3.75 s.

With `max_steps=1` the run ends at time 1.5 with the first four messages: one
model call, both tools executed and answered.

## Constraints

- `max_steps` $\ge 1$. At most 16 tool calls per step and 40 steps. Tool results
  are JSON-serializable.
- The transcript is compared exactly, message by message, and the model must not
  have reported a protocol violation.
- **Cost.** The number of model calls is exactly the number of steps the rule
  gives (never more than `max_steps`), and the tools are run exactly as the table
  says: once per call that reaches the last two rows, with the arguments the model
  gave, and never for a call with an unknown name or invalid arguments.
- **Time.** All times in the tests are multiples of 1/64 s, so they are exact. The
  simulated time at which `run_agent` returns must be exactly the sum, over the
  steps, of the model's latency plus the **longest** tool of the step, and every
  tool of a step must have been started at the instant the model's message arrived.
- A model that is called more often than `max_steps` allows eventually raises; a
  case also fails after 5 seconds of real time.

## Why this matters

Every agent framework has this loop at its center, and every production incident
with agents runs through it. A tool that raises takes the whole request down with
a 500 when the model could have read the error and tried again. A result attached
to the wrong `tool_call_id` is accepted by the API and makes the model reason from
the weather of the wrong city. Skipping one of several parallel calls is rejected
by the API on the next request, so the conversation is stuck. Tools awaited one
after another make a five-way parallel search five times slower than it has to be.
And a model that keeps asking for tools, with no step limit, is an unbounded bill.
