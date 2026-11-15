The loop of `tool-call-loop` waits for the model's whole turn before it starts any
tool. But a turn arrives as a stream, a tool call is known long before the turn
ends, and often it can be guessed before it has been written out. Write the loop
that starts read-only tools early: on a prediction while the arguments are still
streaming, or as soon as the arguments are complete. It must use an early result
only if the model ends up making exactly that call, cancel it otherwise, and never
start a tool with side effects before the turn is over.

## Specification

```python
async def run_agent(
    model, tools: dict, messages: list[dict], predictor, max_steps: int
) -> list: ...
```

The messages, the transcript, the outcome of a call, the order of the tool
messages, and the step limit are those of `tool-call-loop`; they are restated
below. What is new is the stream and the timing.

**The stream.** `model.stream(transcript)` starts one model turn and returns an
async iterator over its *pieces* (`async for piece in model.stream(transcript)`).
The pieces of a turn, in order:

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

Calls are streamed one after another: after the header of call $i + 1$, no more
fragments of call $i$ arrive. `assemble(pieces)` from the kit
(`from whetstone.kit import assemble`) turns the list of all pieces of a turn into
the assistant message, which is appended to the transcript unchanged. The model
validates the conversation when `stream` is called, as in `tool-call-loop`.

**The tools.** `tools` maps names to kit `ScriptedTool`s. `await tool(**arguments)`
runs the tool and takes time. `tool.side_effects` is `False` for a tool that only
reads (it may be started speculatively and abandoned) and `True` for a tool that
changes something. Canceling the task that awaits a tool stops the run.

**The predictor.** `predictor(name, text)` is a plain function. Given a tool's name
and the argument text of a call received so far (`""` when only the header has
arrived), it returns the arguments it expects the call to end up with, as a dict,
or `None` for "no prediction".

**When a call's tool is started.** Three instants matter for call $i$: its header
arrives; it is **complete** when the first piece that does not belong to it
arrives (the header of call $i + 1$, or the finish piece); and the **turn ends**
when the finish piece arrives. With `name` the call's tool name:

1. **Speculation.** If `name` is a key of `tools` and that tool has no side
   effects: when the header arrives, and again after each argument fragment, as
   long as no run has been started for this call, call `predictor(name, text)`
   with the argument text so far. If it returns a dict, start
   `tool(**prediction)` at that instant. This is the call's *speculative run*; the
   predictor is not asked again for this call. For an unknown name or a tool with
   side effects nothing is started.
2. **Completion.** Decide the call by the first row that applies, where *final* is
   the complete argument text parsed as JSON:

   | situation | at completion |
   |---|---|
   | `name` is not a key of `tools` | outcome `{"error": "unknown tool: <name>"}`; nothing runs |
   | the text is not valid JSON, or not a JSON object | discard the speculative run, if any; outcome `{"error": "invalid arguments"}` |
   | there is a speculative run and `prediction == final` | the speculative run **is** the call's run: keep it |
   | the tool has no side effects | discard the speculative run, if any; start `tool(**final)` at this instant |
   | the tool has side effects | start `tool(**final)` when the turn ends, never earlier |

   `prediction == final` is equality of the two Python dicts, so spacing and key
   order in the model's text do not matter and a missing or extra argument does.
3. **Discarding** a speculative run means: cancel it if it is still running, and
   never use its result. A discarded run that had already finished, or already
   failed, has no further effect.
4. **Outcomes.** The outcome of a call whose tool runs is that of
   `tool-call-loop`: the result if it is a `str`, `json.dumps(result)` otherwise,
   and `json.dumps({"error": "tool failed: " + str(exc)})` if the run raises. Error
   outcomes are `json.dumps({"error": ...})` with the texts above.

**The loop.** A step is one turn. Repeat at most `max_steps` times: stream a turn
and append its assistant message. If it has no tool calls, return the transcript.
Otherwise wait until the turn has ended and every run that was kept or started has
finished, append one message
`{"role": "tool", "tool_call_id": call["id"], "content": outcome}` per call in
call order, and start the next turn at that instant. After step `max_steps` the
transcript is returned with that turn's calls answered. `messages` is not
modified. When `run_agent` returns, no tool run may still be in progress.

Use `asyncio` only (tasks, `ensure_future`, `cancel`), never the real clock.

## Example

One piece arrives every 0.125 s, with 8 characters of text per piece.
`get_weather` takes 0.6875 s and `search` 1.0625 s (both read-only); `send_email`
takes 0.4375 s and has side effects. The predictor guesses that the object ends
where the text so far ends: it returns `json.loads(text + "}")` when that is a
dict, else `None`.

Turn 1 requests `get_weather({"city": "Paris"})` and `send_email({"to": "ann"})`;
turn 2 requests `search` with the text `{"q":"x","limit":1}`; turn 3 says `Done.`

| time | piece | what happens |
|---|---|---|
| 0.125 | header of `call_0`, `get_weather` | predictor on `""`: `None` |
| 0.25 | `{"city":` | predictor: `None` |
| 0.375 | ` "Paris"` | predictor: `{"city": "Paris"}`. **`get_weather` starts** |
| 0.5 | `}` | (already speculating) |
| 0.625 | header of `call_1`, `send_email` | `call_0` is complete and equals the prediction: kept. `send_email` has side effects: no prediction |
| 0.75, 0.875 | `{"to": "`, `ann"}` | |
| 1.0 | finish | `call_1` is complete; the turn ends: **`send_email` starts** |
| 1.0625 | | `get_weather` returns `"21 C in Paris"` |
| 1.4375 | | `send_email` returns; two tool messages; turn 2 begins |
| 1.5625 | header of `call_2`, `search` | predictor: `None` |
| 1.6875 | `{"q":"x"` | predictor: `{"q": "x"}`. **`search(q="x")` starts** |
| 1.8125, 1.9375 | `,"limit"`, `:1}` | |
| 2.0625 | finish | final is `{"q": "x", "limit": 1}`, not the prediction: the run is **canceled** and **`search(q="x", limit=1)` starts** |
| 3.125 | | `search` returns `["x result 0"]`; turn 3 begins |
| 3.375 | | `Done.` has arrived (one piece and the finish piece); return |

The run takes 3.375 s. The loop of `tool-call-loop` on the same stream takes
3.625 s: it would start `get_weather` at 1.0 and wait for it until 1.6875. The
wrong guess in turn 2 cost a canceled run and no time. Had `search(q="x")` been
allowed to stand in for the real call, the model would have received two results
where it asked for one.

## Constraints

- `max_steps` $\ge 1$; at most 8 tool calls per turn and 8 turns. The predictor is
  deterministic and does not raise.
- The transcript is compared exactly, and the model must not have reported a
  protocol violation.
- **Tool executions.** For every tool, the set of its runs is compared exactly with
  what the rules prescribe: arguments, start time, how each ended (done, failed, or
  canceled), and when. One run too many, a run that starts late, or a speculative
  run that is not canceled at the completion of its call fails the case. A tool
  with side effects must have no run that starts at any instant other than the end
  of a turn.
- **Time.** The simulated time at which `run_agent` returns must not exceed the
  time the rules give. The tests report the time of the non-speculative loop next
  to it.
- All times are multiples of 1/16 s, so they are exact, and they are chosen so
  that no tool finishes at the same instant as a piece arrives: the order of
  simultaneous events never matters.
- A case fails after 5 seconds of real time.

## Why this matters

In an agent, the model's output tokens and the tools run one after the other, and
the user waits for the sum. Tool calls are long to write and quick to predict: a
model takes a second to spell out a search query whose first words already
determine it, and agents repeat themselves (after every edit, the tests; after
every search, the first result). Starting the tool while the model is still
talking hides tool latency behind generation latency, and across many steps that
is a large share of the total time. The danger is exactly proportional to the
gain. A speculative result that is used although the model asked for something
slightly different is a wrong answer that no log explains. A speculative run that
is not canceled keeps a database connection or a browser busy for nothing. And a
speculative `send_email` or `delete_file` cannot be taken back, which is why the
side-effect flag is not an optimization hint but the line between a cache and an
incident.
