Look at one call on a timeline: its header arrives, its argument fragments arrive,
then the first piece that is not its own arrives, and later the turn ends. At
which of these instants do you know the tool? A guess of the arguments? The
arguments for certain? That the model really finished the turn? Now ask, for a
tool that only reads and for a tool that writes, which of those certainties you
need before you start it.

---

Keep one small record per call: its name, the argument text so far, the task of
the run it will use (if one has been started), and the prediction that task was
started with. Start tools with `asyncio.ensure_future(tool(**arguments))` so that
they run while you keep reading the stream; do not `await` a tool inside the
`async for`. Handle a piece in this order: if it is a header, first *complete* the
previous call, then open the new one; add the fragment to the text; then
*speculate* on the current call. On the finish piece, complete the last call and
start everything you held back for the end of the turn. Only after the stream has
ended, await the tasks in call order.

---

```
speculate(call):
    tool = tools.get(call.name)
    if tool is None or tool.side_effects or call.task is not None: return
    guess = predictor(call.name, call.text)
    if guess is not None: call.guess, call.task = guess, ensure_future(tool(**guess))

complete(call):
    final = parsed arguments, or None if not a JSON object
    if call.task and (final is None or call.guess != final):
        call.task.cancel(); call.task = None          # discard: never use it
    if unknown tool:        call.error = "unknown tool: ..."
    elif final is None:     call.error = "invalid arguments"
    elif call.task is None:                           # no confirmed speculation
        if tool.side_effects: at_turn_end.append((call, tool, final))
        else:                 call.task = ensure_future(tool(**final))
```

The outcome of a call is its `error`, or the awaited `call.task` (a raised
exception becomes the `tool failed` outcome). Compare dicts, not strings:
`{"limit": 1, "q": "x"}` and `{"q":"x","limit":1}` are the same call.
