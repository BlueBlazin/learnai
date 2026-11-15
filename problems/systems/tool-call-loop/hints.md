Look at the conversation from the model's side. After an assistant message with
three tool calls, what must the next three messages be before the model will accept
the conversation again? Now ask the same question when the second call names a tool
you do not have, or when its tool raised. Is there any situation in which fewer
than three tool messages is the right answer?

---

Write one coroutine that turns a single tool call into its outcome string and
**never raises**: unknown name, `json.loads` failing, arguments that are not a
dict, and an exception from the tool each `return` an error string. With that, a
step is simple: start that coroutine for every call, wait for all of them, and
append one tool message per call. Keep the loop a `for _ in range(max_steps)`
with the `return` for a final answer inside it, so the limit cannot be forgotten.

---

`asyncio.gather` does the concurrency and the ordering in one line:

```python
outcomes = await asyncio.gather(*(execute(tools, call) for call in calls))
for call, outcome in zip(calls, outcomes, strict=True):
    transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})
```

It starts the coroutines in argument order and returns their results in argument
order, however long each took. Because `execute` catches its own exceptions,
`gather` never sees one, and one failing tool cannot cancel the others. Work on
`transcript = list(messages)`, not on `messages`.
