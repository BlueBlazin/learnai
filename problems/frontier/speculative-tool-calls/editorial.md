## The idea

A turn that calls tools has two latencies in series: the model writes the call,
then the tool runs. Speculation overlaps them. What makes it safe is to separate
three questions that the non-speculative loop answers all at once, at the end of
the turn.

**What will the call be?** Before the arguments are complete this is a guess, and
the predictor supplies it. A guess costs nothing to make and something to act on,
so it is acted on only where acting is free of consequences: a read-only tool.

**What is the call?** When the next call's header or the finish piece arrives, the
arguments are final. Now a speculative run is either the call the model made or it
is not; there is no "close". If the dicts are equal, the run that is already under
way *becomes* the real run and the time since it started is the gain. If not, it
is canceled and the real call starts. And a read-only call that nobody predicted
can still start now rather than at the end of the turn: with several calls in a
turn, every call but the last is complete while the model is still writing the
others.

**Did the turn happen?** A turn can still be cut off after a call is complete: the
token limit, a disconnect, a guardrail that rejects the message. For a read-only
tool that does not matter. For a tool with side effects it is the whole question,
so those calls wait for the finish piece, whatever is known earlier.

The implementation follows from keeping the tools off the reading path. Each run
is a task; reading the stream never waits for a tool; and only when the stream
has ended does the loop await the tasks, in call order, to build the tool
messages.

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

`outcome_of` returns `call.error`, or awaits `call.task` and maps a result or an
exception to a string as in `tool-call-loop`. Awaiting the tasks one after another
does not serialize them: they have been running since they were created.
`discard` retrieves the exception of a run that had already failed because asyncio
reports a task whose exception nobody looked at; it is good hygiene, and the tests
do not depend on it.

## Common mistakes

- **Not speculating, or speculating halfway.** The loop of `tool-call-loop` gives
  the right transcript and all of the latency. Using predictions but starting
  unpredicted read-only calls only at the end of the turn leaves the cheapest gain
  on the table: those calls were certain, not guessed.
- **Speculating on side effects.** Asking the predictor for every tool sends the
  email at the first header. Canceling it later does not unsend it. The related
  mistake is to treat "arguments complete" as "safe to run": for a tool with side
  effects the line is the end of the turn.
- **A near match counted as a match.** Same tool, or all predicted arguments
  present in the final ones, is not the call the model made: `search(q)` returns
  two results where `search(q, limit=1)` returns one, and the model receives an
  answer to a question it did not ask.
- **Comparing text.** `json.dumps(prediction) == text` rejects correct predictions
  whenever the model spaces or orders its JSON differently; the speculation is
  paid for and thrown away.
- **Not canceling.** A discarded run that keeps going holds its backend, and when
  the loop returns it is still running. The same happens when a call turns out to
  have malformed arguments and its speculative run is forgotten rather than
  discarded.
- **Using a wrong guess because it has finished.** That the result is already
  there does not make it the result of the call.
- **Restarting a confirmed call.** Canceling the speculative run and starting the
  same call again is correct and gains nothing.
- **Asking the predictor again while a run is under way.** Every new answer starts
  another run; a turn with long arguments starts a dozen.
- **Results in completion order.** Speculation makes completion order even less
  related to call order than in the plain loop. The tool messages follow the calls.

## Complexity

Per turn the work is linear in the number of pieces, with one predictor call per
piece of a read-only call until its first prediction. Each call starts at most two
runs (one speculative, one real). The latency of a turn is the later of the end of
the stream and the last tool to finish, where a confirmed prediction moves a
tool's start from the end of the turn to the instant of the prediction, and an
unpredicted read-only call to the instant its arguments are complete. A wrong
prediction costs one canceled run and no latency.

## In the wild

- **This problem is a design exercise**, informed by the sources below; its rules
  (the predictor's signature, asking it at the header and after each fragment,
  one speculative run per call, exact equality for confirmation, read-only calls
  starting at completion, side effects at the end of the turn) are this problem's
  own and do not describe a particular system.
- **Speculating on tool calls during generation** is the subject of "Parallelizing
  Tool Execution and LLM Generation for Low-Latency Agent Serving" (arXiv
  2603.18897), which presents a system called PASTE. Per its abstract, PASTE
  predicts concrete future tool invocations from recurring agent patterns, executes
  them ahead of time while the model is still generating, keeps speculative results
  isolated until the model confirms them, and reports a 43.5 percent lower average
  task completion time. The predictor here stands in for that pattern-based
  prediction, and "use the result only if the model makes exactly that call" for
  the confirmation. The abstract does not say how tools with side effects are
  handled; the rule here is this problem's.
- **Running the calls of a plan in parallel** rather than one model step at a time
  is the idea of "An LLM Compiler for Parallel Function Calling" (arXiv 2312.04511,
  ICML 2024), in which a planner emits a graph of calls and an executor dispatches
  each as soon as its dependencies are met. Starting a call when its own arguments
  are complete, without waiting for the rest of the turn, is the streaming-loop
  version of that.
- **What is simplified.** Real tools are rarely purely read-only or purely
  writing: reads cost money, hit rate limits, and show up in audit logs, so a real
  system needs a budget for speculation as well as a flag. Real predictors return
  several candidates with confidences. And a turn here always ends normally; the
  reason side effects wait for the finish piece is the turn that does not.
