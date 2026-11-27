Think about the conversation from the model's point of view. After an assistant message requests three tool calls, what must the next three messages in the transcript be before the model will accept the conversation again? Does that requirement change if one call names an unknown tool or raises an exception?

---

Write a helper coroutine that turns a single tool call into its outcome string and **never raises**. Unknown tool names, `json.JSONDecodeError`, non-dict JSON arguments, and exceptions from `await tool(**arguments)` should all return a JSON error string. With that helper in place, run all calls in a step concurrently, wait for all of them to finish, and append one tool message per call. Use `for _ in range(max_steps):` with an early `return` on a final answer so the step limit is enforced automatically.

---

`asyncio.gather` handles both concurrency and ordering in one call:

```python
outcomes = await asyncio.gather(*(execute(tools, call) for call in calls))
for call, outcome in zip(calls, outcomes, strict=True):
    transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})
```

`gather` starts the coroutines in argument order and returns their outcomes in argument order regardless of which finishes first. Because `execute` catches its own exceptions, one failing tool cannot cancel its siblings. Initialize `transcript = list(messages)` so you do not mutate the caller's list.
