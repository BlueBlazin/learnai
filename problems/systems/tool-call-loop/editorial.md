## The idea

The loop has one invariant, and every rule of the problem follows from it: **the
transcript is a valid conversation whenever the model is called, and whenever the
function returns.** For the chat-completions protocol, valid means that each
assistant message with tool calls is followed directly by one tool message per
call, matched by `tool_call_id`.

From the invariant:

- *Every call gets an answer.* A call to a tool that does not exist, a call whose
  arguments do not parse, and a call whose tool raised still need a tool message.
  The only thing left to decide is its content, and the useful content is the
  error itself: the model reads `{"error": "invalid arguments"}` and emits the
  call again, correctly. Raising instead ends a conversation the model could have
  repaired.
- *Ids come from the calls.* The tool message for a call carries that call's `id`,
  read from the call. Anything else (a counter, the position, the order in which
  results arrived) is a second source of truth that will disagree with the first.
- *The step limit stops model calls, not tool messages.* If the last allowed step
  requested tools, they are run and answered before returning; the caller gets a
  transcript it can hand back to continue.

Concurrency is a separate concern and it composes cleanly if the per-call work is
a coroutine that cannot fail. `execute(call)` returns an outcome string in all four
situations of the table. Then "run the calls of a step concurrently, keep the
outcomes in call order, let no failure affect a sibling" is `asyncio.gather` with
no further thought: gather starts its arguments in order, returns results in
argument order, and has no exception to propagate.

## Reference solution (Python)

```python
async def run_agent(model, tools, messages, max_steps):
    transcript = list(messages)

    for _ in range(max_steps):
        reply = await model.acomplete(transcript)
        transcript.append(reply)
        calls = reply.get("tool_calls")
        if not calls:
            return transcript

        outcomes = await asyncio.gather(*(execute(tools, call) for call in calls))
        for call, outcome in zip(calls, outcomes, strict=True):
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})

    return transcript


async def execute(tools, call):
    function = call["function"]
    tool = tools.get(function["name"])
    if tool is None:
        return error(f"unknown tool: {function['name']}")

    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return error("invalid arguments")
    if not isinstance(arguments, dict):
        return error("invalid arguments")

    try:
        result = await tool(**arguments)
    except Exception as exc:
        return error(f"tool failed: {exc}")
    return result if isinstance(result, str) else json.dumps(result)


def error(text):
    return json.dumps({"error": text})
```

`except Exception` is deliberate, and so is not writing `except BaseException`:
`asyncio.CancelledError` is a `BaseException`, and a canceled agent must stop, not
report "tool failed" to a model nobody is waiting for.

## Common mistakes

- **Executing only the first tool call.** `reply["tool_calls"][0]` works on every
  single-call example. With two calls, the next request is rejected: the second id
  has no tool message.
- **Letting a tool's exception escape.** One flaky tool ends the run. The same
  goes for `tools[name]` on an unknown name (`KeyError`) and `json.loads` on
  truncated arguments (`JSONDecodeError`).
- **Passing non-object arguments on.** `json.loads('[1, 2]')` succeeds, and
  `tool(**[1, 2])` raises a `TypeError` that the tool's `except` then reports as
  "tool failed". The tool never failed: it was never called.
- **Pairing results with ids by arrival.** Collecting outcomes with
  `asyncio.as_completed` (or appending from inside each task) puts them in
  completion order. Zipped with the calls, the fast tool's result lands under the
  slow tool's id. The API accepts it; the model is now wrong about the world.
- **Inventing ids.** `f"call_{n}"` happens to match one provider's numbering on a
  good day.
- **Awaiting the calls one after another.** Correct transcript, and a step takes
  the sum of the tool latencies instead of the maximum.
- **One failure fails the step.** `gather` without a per-call `try` raises the
  first exception; catching it around the whole step reports the failure for
  every call, including those that succeeded (and whose side effects happened).
- **No step limit, or a limit that is off by one** (`while True` with a counter
  checked after the call).
- **Stopping at the limit before answering.** Returning right after the last
  model call leaves an assistant message with open tool calls at the end of the
  transcript, which no API will accept as history.
- **Growing the caller's list.** `transcript = messages` appends to the list the
  caller passed in; a caller that retries then sends the previous attempt along.
- **Encoding strings again.** `json.dumps("21 C in Paris")` adds quotes and
  escapes; a string result is already the content.

## Complexity

With $s$ steps and $c_i$ calls in step $i$: $s \le$ `max_steps` model calls and
$\sum_i c_i$ tool runs at most. In time, a step costs the model's latency plus
$\max$ (not $\sum$) of its tools' latencies. The conversation grows with every
step and is sent whole each time, so the model's input over a run is quadratic in
the number of steps: the reason prefix caching matters for agents.

## In the wild

- The message format is that of OpenAI-compatible chat completions: an assistant
  message carries `tool_calls`, each with an `id`, `type: "function"`, and a
  `function` with a `name` and `arguments` as a JSON-formatted string; the result
  goes back as a message with `role: "tool"`, the `tool_call_id` it responds to,
  and `content` (OpenAI API reference, "Create chat completion"). The function
  calling guide says to assume that a response may contain several calls and to
  handle each one, and that a tool's output is a string in a format of your
  choice, which is where error results go; there is no separate error channel.
- Agent frameworks expose the step limit under names such as `max_turns` or
  `max_iterations`; whatever it is called, check that yours has one.
- `speculative-tool-calls` (Frontier track) builds on this loop: it starts tools
  before the model has finished writing their arguments.
