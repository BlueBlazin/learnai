## The idea

Every requirement in this problem follows from one invariant: **the transcript must be a valid chat-completions conversation both before every model call and when `run_agent` returns.** In the chat-completions protocol, a conversation is valid only if every assistant message containing `"tool_calls"` is immediately followed by one tool message per call, matched by `"tool_call_id"`.

Three design rules follow directly from that invariant:

- **Every call gets a tool message.** Unknown tool names, malformed or non-object JSON arguments, and exceptions raised by tools must all produce a tool message rather than crashing the loop. Returning the error as JSON lets the model see what went wrong (such as `{"error": "invalid arguments"}`) and try again on the next step.
- **Copy IDs directly from each call.** Always read `call["id"]` from the call dict itself. Generating synthetic IDs or pairing results by completion order creates a second source of truth that easily drifts out of sync.
- **The step limit caps model calls, not tool responses.** If the model requests tools on step `max_steps`, execute them and append their tool messages before returning. Leaving dangling tool calls at the end of the transcript produces a history that no API will accept on a follow-up turn.

Concurrency stays clean when you encapsulate a single call in a coroutine that never raises. Write `execute(tools, call)` so that all four rows of the outcome table return a string. Once `execute` cannot raise, `asyncio.gather` handles the rest automatically: it starts coroutines in argument order, runs them concurrently, returns outcomes in argument order, and never cancels sibling tasks on a tool failure.

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

Catching `Exception` instead of `BaseException` is intentional. `asyncio.CancelledError` inherits from `BaseException`, so if the caller cancels the agent task, it propagates cleanly instead of being turned into a `"tool failed"` message.

## Common mistakes

- **Executing only the first tool call.** Indexing `reply["tool_calls"][0]` passes single-call tests, then fails protocol validation as soon as the model emits two calls in one step.
- **Letting errors escape.** Uncaught `KeyError` on `tools[name]`, `json.JSONDecodeError` on truncated arguments, or tool exceptions abort a run that the model could have recovered from.
- **Unpacking non-object JSON arguments.** `json.loads('[1, 2]')` succeeds, and passing it to `tool(**arguments)` raises a `TypeError`. If that `TypeError` is caught by the tool's `except` block, it reports `"tool failed"` even though the tool was never valid to call.
- **Pairing results with IDs by arrival order.** Using `asyncio.as_completed` or appending from inside each task orders outcomes by completion time. Zipping those with `calls` attaches the faster tool's output to the slower tool's ID, which passes API validation while silently corrupting the model's context.
- **Synthesizing call IDs.** Formatting `f"call_{n}"` happens to match some providers' IDs in simple cases, then breaks whenever a model uses arbitrary IDs.
- **Awaiting calls sequentially.** Looping with `await execute(...)` produces the right transcript, but the step takes the sum of the tool latencies instead of the maximum.
- **Catching exceptions around `asyncio.gather` instead of inside each call.** Without a per-call `try`, the first tool exception aborts `gather`, losing the results of sibling calls that succeeded.
- **Missing or off-by-one step limit.** A `while True` loop that checks the step counter after the model call allows `max_steps + 1` calls.
- **Stopping at `max_steps` before answering tool calls.** Returning immediately after the final model call leaves unanswered tool calls at the end of `transcript`.
- **Mutating the caller's `messages` list.** Writing `transcript = messages` modifies the caller's list in place, polluting subsequent retries.
- **Double-encoding string results.** Calling `json.dumps("21 C in Paris")` wraps the string in extra quotes and escapes instead of passing it through unchanged.

## Complexity

For a run of $s \le \text{max\_steps}$ steps with $c_i$ tool calls in step $i$, the loop makes $s$ model calls and at most $\sum_{i=1}^{s} c_i$ tool executions. Each step takes the model's latency plus the **maximum** (not the sum) of its tools' latencies. Because the full transcript is sent on every step and grows with each turn, total input tokens across a run grow quadratically with $s$, which is why prompt prefix caching is essential for agent workloads.

## In the wild

- This message format matches OpenAI-compatible chat completions ("Create chat completion" in the OpenAI API reference). Assistant messages carry `"tool_calls"` with `"id"`, `"type": "function"`, and `"function"` (`"name"` and JSON-encoded `"arguments"`), and each response is sent back with `"role": "tool"`, `"tool_call_id"`, and a string `"content"`. Because there is no separate error channel in the protocol, tool errors are serialized into `"content"` for the model to read.
- Production agent frameworks expose `max_steps` under names like `max_turns` or `max_iterations`. Always verify that a hard bound is configured before deploying an agent loop.
- The `speculative-tool-calls` problem (Frontier track) extends this loop by launching tools before the model finishes streaming their arguments.
