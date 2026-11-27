## The idea

A turn with tool calls puts two latencies in series, since the model streams the call and then the tool runs. Speculation overlaps them. To keep that overlap safe, separate three questions that a non-speculative loop answers all at once at the end of the turn.

**What will the call be?** While arguments are still streaming, we only have a guess from the predictor. Because acting on a guess can be wrong, we only speculate on read-only tools where abandoning a run has no side effects.

**What is the final call?** Once the next call's header or the finish piece arrives, the current call's arguments are final. At that point, any speculative run either matches the final dict exactly or does not. On an exact match, the in-flight speculative run becomes the real run and we save all the time since it started. On a mismatch, we cancel the speculative run and launch the real call. Even when no prediction was made, an unpredicted read-only call can start right at completion rather than waiting for the rest of the turn to stream.

**Did the turn complete?** Even after one call's arguments finish streaming, the turn can still be cut off by a token limit, a network disconnect, or a guardrail rejection. Read-only tools can ignore that risk, whereas tools with side effects must wait for the finish piece before starting.

The implementation stays clean by keeping tool execution off the stream-reading path. Launch each tool as an asyncio task without awaiting it inside the stream loop. Only after the stream ends do you await the tasks in call order to assemble the tool messages.

## Reference solution (Python)

```python
class Call:
    def __init__(self, name):
        self.name, self.text = name, ""
        self.task = self.guess = self.error = None


async def run_agent(model, tools, messages, predictor, max_steps):
    transcript = list(messages)
    for _ in range(max_steps):
        message, outcomes = await run_turn(model.stream(transcript), tools, predictor)
        transcript.append(message)
        calls = message.get("tool_calls")
        if not calls:
            return transcript
        for call, outcome in zip(calls, outcomes, strict=True):
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})
    return transcript


async def run_turn(stream, tools, predictor):
    pieces, calls, at_turn_end = [], [], []

    def speculate(call):
        tool = tools.get(call.name)
        if tool is None or tool.side_effects or call.task is not None:
            return
        guess = predictor(call.name, call.text)
        if guess is not None:
            call.guess, call.task = guess, asyncio.ensure_future(tool(**guess))

    def complete(call):
        tool = tools.get(call.name)
        arguments = parse(call.text)  # a dict, or None
        if call.task is not None and (arguments is None or call.guess != arguments):
            discard(call.task)
            call.task = None
        if tool is None:
            call.error = error(f"unknown tool: {call.name}")
        elif arguments is None:
            call.error = error("invalid arguments")
        elif call.task is None:
            if tool.side_effects:
                at_turn_end.append((call, tool, arguments))
            else:
                call.task = asyncio.ensure_future(tool(**arguments))

    async for piece in stream:
        pieces.append(piece)
        for part in piece.get("tool_calls", ()):
            function = part.get("function", {})
            if part["index"] == len(calls):  # a header: the call before it is complete
                if calls:
                    complete(calls[-1])
                calls.append(Call(function["name"]))
            calls[-1].text += function.get("arguments", "")
            speculate(calls[-1])
        if "finish_reason" in piece:
            if calls:
                complete(calls[-1])
            for call, tool, arguments in at_turn_end:
                call.task = asyncio.ensure_future(tool(**arguments))

    return assemble(pieces), [await outcome_of(call) for call in calls]


def discard(task):
    if not task.cancel():  # it had finished, and may hold an exception
        task.exception()
```

`outcome_of` returns `call.error` when present, or awaits `call.task` and converts its result or exception into a string as in `tool-call-loop`. Awaiting the tasks sequentially after the loop does not serialize them because every task is already running. In `discard`, calling `task.exception()` on an already-finished task clears any unhandled exception warning from `asyncio`, which is good hygiene even though the tests do not require it.

## Common mistakes

- **Not speculating, or speculating halfway.** Using the plain `tool-call-loop` produces the right transcript with none of the latency reduction. Similarly, running predictions while delaying unpredicted read-only calls until the turn ends misses a guaranteed speedup because those calls are already complete.
- **Speculating on side effects.** Calling the predictor for every tool can send an email as soon as the header arrives, and canceling the task later cannot undo it. Treating "arguments complete" as safe for side-effecting tools makes the same mistake, since those tools must wait for the turn to end.
- **Accepting a near match.** Matching only the tool name or checking that predicted keys are a subset of the final arguments returns the wrong result. For example, `search(q)` may return two results when the model asked for `search(q, limit=1)`.
- **Comparing raw JSON strings.** Checking `json.dumps(prediction) == text` rejects valid predictions whenever the model uses different whitespace or key ordering, wasting the speculative run.
- **Not canceling discarded runs.** Leaving a mismatched speculative task running keeps its backend busy and violates the invariant that no runs remain active when `run_agent` returns. The same leak happens if a call's final arguments are malformed JSON and its speculative task is forgotten instead of canceled.
- **Keeping a wrong guess because it already finished.** Having a completed result ready does not make it the result of the call the model actually made.
- **Restarting a confirmed call.** Canceling a matching speculative run and starting an identical call again produces the right transcript but throws away the entire latency gain.
- **Calling the predictor again while speculating.** Querying the predictor on subsequent fragments after a speculative run has started launches extra redundant runs.
- **Returning results in completion order.** Speculation makes tool completion order even more decoupled from call order than in a plain concurrent loop. Tool messages must still be appended in call order.

## Complexity

Per turn, the loop does work linear in the number of streamed pieces, calling the predictor at most once per piece of a read-only call until its first prediction. Each call starts at most two tool runs (one speculative and one real). Turn latency is the later of the stream end and the slowest tool run. A confirmed prediction moves a tool's start time from the end of the turn to the prediction instant, and an unpredicted read-only call starts as soon as its arguments finish streaming. A wrong prediction costs one canceled run and adds no latency.

## In the wild

- **Design scope.** This problem is a design exercise informed by the systems below. Its specific protocol (the predictor signature, querying at the header and each fragment, at most one speculative run per call, exact dict equality, starting read-only calls at completion, and holding side effects until turn end) is tailored to this problem rather than copied from a single production codebase.
- **Speculative tool execution during generation.** "Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving" (arXiv 2603.18897) introduces PASTE. According to its abstract, PASTE predicts concrete tool invocations from recurring agent patterns, executes them ahead of time while the model is generating, isolates speculative results until the model confirms them, and reduces average task completion time by 43.5 percent. The predictor here models that pattern-based prediction, and exact dict equality models confirmation. The abstract does not specify how side effects are handled, so the turn-end rule here is this problem's own.
- **Parallel call dispatch.** "An LLM Compiler for Parallel Function Calling" (arXiv 2312.04511, ICML 2024) uses a planner to emit a dependency graph of tool calls and an executor to dispatch each call as soon as its dependencies are satisfied. Starting a read-only call as soon as its arguments finish streaming is the streaming-loop analogue of that idea.
- **Simplifications.** Real tools are rarely purely read-only or purely mutating. Even read calls cost money, consume rate limits, and appear in audit logs, so production systems pair a side-effect flag with a speculation budget. Real predictors also return multiple ranked candidates with confidence scores. Finally, turns in these tests always finish cleanly, whereas real systems hold side effects until the finish piece precisely because a turn can abort mid-stream.
