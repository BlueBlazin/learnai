## The idea

The prompt of a chat request has three parts with different rights:

- the **system message**, which configures the assistant and must survive,
- the **last turn**, which is the request itself,
- the **history** in between, which is context: useful, and the only part that can be
  given up.

History is given up oldest first, because recent turns are what the next answer most
likely refers to, and in whole **turns**, because a turn is the smallest unit that
makes sense on its own. An assistant message without its question, or a tool result
without the call that produced it, is worse than nothing: the first confuses the
model, and the second is a malformed prompt for many chat templates.

A turn starts at each user message. So the legal cut points are exactly the indices of
the user messages, and the candidates are "system message plus everything from user
message $i$ on". The answer is the first candidate, in order of decreasing length,
that satisfies

$$
\texttt{count\_tokens(kept)} + \texttt{max\_new\_tokens} \le \texttt{max\_context} .
$$

Both sides of that inequality hold a trap. On the left, the count must be the count of
the prompt as the template renders it. Templates add tokens per prompt (BOS, the
generation prompt, a default system turn) and merge messages (consecutive tool
results), so per-message counts do not add up. On the right, the window is shared
between prompt and answer: a prompt that fills it leaves the model no room to
generate.

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

The last cut point is the latest user message, so the final candidate is the mandatory
part, and falling out of the loop means even that does not fit. Whatever follows the
latest user message (an assistant message with tool calls, tool results) belongs to
the last turn and is kept with it without any extra code.

## Common mistakes

- **No reservation.** Comparing the prompt with `max_context` alone. Everything works
  until a conversation gets long, and then answers stop after a few tokens with finish
  reason `length`.
- **Dropping the system message**, by treating the conversation as a flat list and
  cutting from the front.
- **Cutting inside a turn**: one message at a time, or two at a time on the assumption
  that user and assistant alternate strictly. Tool calls break the assumption.
- **`<` for `<=`.** A prompt that exactly fills the remaining window fits.
- **Adding per-message counts.** Wrong by the template's per-prompt tokens, in either
  direction depending on how the per-message count is taken.
- **Returning the mandatory part when it does not fit.** The request cannot be served
  as asked; the caller has to know (to return an error, or to truncate the user's
  message deliberately). Silently sending an over-long prompt moves the failure into
  the engine.
- **Dropping from the wrong end**, keeping the opening of the conversation and losing
  the exchange right before the question.
- **Trimming the caller's list in place.** The application's history is also what it
  shows the user and stores; the trimmed prompt is a view of it for one request.

## Complexity

With $T$ turns and $n$ messages, at most $T$ calls of `count_tokens`, each of which
renders and tokenizes up to the whole conversation: $O(T \cdot n)$ message
tokenizations in the worst case. That is fine for a chat (and the common case, where
everything fits, is one call). A server that trims on every request caches the token
count per message and the template's fixed overhead, or, when the count is monotone
in the number of kept turns (dropping a turn never makes a prompt longer), finds the
cut by binary search in $O(\log T)$ counts.

## In the wild

Chat front ends and agent frameworks ship this policy as a utility; LangChain's
`trim_messages`, for example, takes a token counter, a budget, `include_system`, and
`start_on="human"`, which are the system-message and whole-turn rules of this problem.
On the engine side the same inequality is enforced once more as request validation:
an OpenAI-compatible server compares the prompt's token count plus the requested
`max_tokens` with the model's context length and rejects the request when the sum is
larger. Trimming also interacts with prefix caching: dropping the oldest turn changes
the prompt right after the system message, so everything cached beyond that point is
recomputed. That is an argument for trimming in large steps (dropping several turns
at once, down to a lower watermark) instead of one turn on every request.
