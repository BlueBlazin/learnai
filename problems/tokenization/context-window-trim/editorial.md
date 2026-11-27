## The idea

A chat prompt has three parts with different roles:

- The **system message** configures the assistant and must survive.
- The **last turn** is the active request and must survive.
- The **history** in between provides context and is the only part that can be dropped.

History is dropped oldest first because recent turns are most likely relevant to the next reply. Turns are dropped whole because a turn is the smallest self-contained unit. An assistant reply without its question confuses the model, and a tool result without the call that produced it yields a malformed prompt under many chat templates.

Every turn begins at a `"user"` message. The legal cut points are therefore the indices of the `"user"` messages. Each candidate `kept` consists of the optional system message plus everything from a `"user"` message at index $i$ onward. We return the first candidate `kept`, in order of decreasing length, that satisfies

$$
\texttt{count\_tokens(kept)} + \texttt{max\_new\_tokens} \le \texttt{max\_context} .
$$

Both sides of this inequality hide a common bug. On the left, the token count must come from rendering the candidate list as a whole. Chat templates add per-prompt tokens (such as BOS, the generation prompt, or a default system turn) and may merge adjacent tool results, so per-message counts do not add up. On the right, the context window is shared between the prompt and the generated reply. A prompt that fills `max_context` leaves the model no room to answer.

## Reference solution (Python)

```python
def trim_messages(messages, count_tokens, max_context, max_new_tokens):
    head = messages[:1] if messages[0]["role"] == "system" else []
    body = messages[len(head) :]
    turn_starts = [i for i, message in enumerate(body) if message["role"] == "user"]
    budget = max_context - max_new_tokens
    for start in turn_starts:
        kept = head + body[start:]
        if count_tokens(kept) <= budget:
            return kept
    raise ValueError("the system message and the last turn do not fit in the context window")
```

The final cut point is the latest `"user"` message, so the last iteration checks the mandatory portion. If the loop finishes without returning, even that mandatory portion does not fit. Any assistant or tool messages after the latest `"user"` message are included automatically as part of `body[start:]`.

## Common mistakes

- **No output reservation.** Comparing `count_tokens(kept)` against `max_context` alone works on short chats, then truncates answers after a few tokens with finish reason `length` once conversations grow long.
- **Dropping the system message.** Treating `messages` as a flat list and slicing from index 0 removes the system prompt.
- **Cutting inside a turn.** Dropping one message at a time, or two at a time assuming strict user/assistant alternation, breaks whenever tool calls and tool results appear.
- **Using `<` instead of `<=`.** A prompt that fills the remaining budget exactly still fits.
- **Adding per-message counts.** Summing individual message counts misses per-prompt template overhead and can be off in either direction depending on how each message is counted.
- **Returning the mandatory part when it does not fit.** When the system message and last turn already exceed the budget, the caller must be notified via `ValueError` so it can return an error or truncate the user's message deliberately. Silently returning an over-long prompt pushes the failure into the inference engine.
- **Dropping from the wrong end.** Keeping the opening turns discards the recent context immediately before the latest question.
- **Modifying the caller's list in place.** The caller also displays and stores the full conversation history. Trimming must return a new list without mutating `messages`.

## Complexity

With $T$ turns and $n$ total messages, the loop calls `count_tokens` at most $T$ times. Each call renders and tokenizes up to all $n$ messages, giving $O(T \cdot n)$ message tokenizations in the worst case. In the common case where the full conversation fits, `count_tokens` runs only once. Servers that trim on every request often cache per-message token counts alongside the template's fixed overhead. When token counts are monotonic in the number of kept turns, binary search can also find the cut point in $O(\log T)$ calls.

## In the wild

Chat front ends and agent frameworks provide this policy as a standard utility. LangChain's `trim_messages`, for example, takes a token counter, a budget, `include_system`, and `start_on="human"`, matching the system-message and whole-turn rules here. On the serving side, OpenAI-compatible engines enforce the same inequality during request validation, rejecting any request where the prompt length plus `max_tokens` exceeds the model's context length. Trimming also affects KV prefix caching. Dropping the oldest turn changes the prompt right after the system message, which invalidates all cached prefixes beyond that point. For this reason, production systems often trim in larger chunks down to a lower watermark rather than dropping a single turn on every request.
