Trace a single tool call along a timeline: its header arrives, its argument fragments arrive, the first piece not belonging to it arrives, and finally the turn ends. At each point, ask what you know: the tool name, a guess at the arguments, the exact final arguments, or that the turn actually finished. Then decide which level of certainty a read-only tool needs versus a tool with side effects.

---

Keep a small record per call storing its name, accumulated argument text, active task (if any), and the prediction used to start that task. Launch tools with `asyncio.ensure_future(tool(**arguments))` so they run concurrently while you keep reading the stream, and never `await` a tool inside the `async for` loop. For each incoming piece, process events in order. If it is a new call header, complete the previous call and open the new one. Append any argument fragment to the current call's text, then try speculating on it. When the finish piece arrives, complete the last call and start any side-effecting calls queued for turn end. Only after the stream finishes should you await the tasks in call order.

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

A call's outcome is its `error` if set, or the awaited `call.task` (converting any raised exception into the `tool failed` error payload). Compare parsed dicts rather than raw JSON strings so `{"limit": 1, "q": "x"}` and `{"q":"x","limit":1}` match.
