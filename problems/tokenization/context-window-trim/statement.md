A chat conversation grows by multiple messages on every exchange, while a model's context window stays fixed. Before sending a request, a chat server trims older history so both the prompt and the model's reply fit in the window. Implement this policy by always keeping the system message and the latest turn, and dropping the oldest turns whole until the prompt fits.

![This problem is the Trim to context step, dropping the oldest turns until the prompt fits.](assets/map.svg)

## Specification

```python
def trim_messages(
    messages: list[dict[str, str]],
    count_tokens: Callable[[list[dict[str, str]]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[dict[str, str]]: ...
```

- `messages`: the conversation in chronological order (oldest first). Each message is a `dict` with keys `"role"` (`"system"`, `"user"`, `"assistant"`, or `"tool"`) and `"content"`.
- `count_tokens(msgs)`: returns the token count of the prompt rendered by the model's chat template for `msgs`, including special tokens, role headers, and the generation prompt. It accepts any list of messages taken from `messages`. Treat it as a black box, because the token count of a list is **not** the sum of the counts of its individual messages.
- `max_context`: the model's total context window, in tokens.
- `max_new_tokens`: the number of tokens reserved for the model's reply.

**Conversation structure.** If a system message is present, it is `messages[0]`, and there is only one. The remaining messages form a sequence of **turns**. A turn starts at a `"user"` message and contains every message up to, but not including, the next `"user"` message (such as the assistant's reply, tool calls, and tool results). The first message after an optional system message is always a `"user"` message, so every non-system message belongs to exactly one turn. The **last turn** starts at the latest `"user"` message. It may consist of that message alone, or it may also include assistant and tool messages when the model is called again in the middle of a tool loop.

**Fitting condition.** A list of messages `kept` fits when

$$
\texttt{count\_tokens(kept)} + \texttt{max\_new\_tokens} \le \texttt{max\_context} .
$$

**Trimming policy.** Let $T$ be the number of turns in `messages`. For $k = 0, 1, \dots, T - 1$, let candidate $k$ be the system message (if present) followed by all turns except the $k$ oldest, in their original order. Return candidate $k$ for the **smallest** $k$ that fits:

- If the whole conversation fits, return it unchanged ($k = 0$).
- Drop turns whole and oldest first. Dropping a turn removes all of its messages, so the result never starts with an `"assistant"` or `"tool"` message after the system message.
- Never drop the system message or the last turn. If even candidate $T - 1$ does not fit, raise `ValueError`.

Return a new list holding the kept messages in order (the same dicts or equal copies). Do not modify `messages` or any message dict, even when raising `ValueError`.

## Example

In this example, a prompt costs 3 fixed tokens plus, for each message, 4 overhead tokens and 1 token per word in its content:

```python
def count_tokens(msgs):
    return 3 + sum(4 + len(m["content"].split()) for m in msgs)


messages = [
    {"role": "system", "content": "Be brief."},  # 6
    {"role": "user", "content": "What is BPE?"},  # 7
    {"role": "assistant", "content": "A subword tokenizer built from merges."},  # 10
    {"role": "user", "content": "And WordPiece?"},  # 6
    {"role": "assistant", "content": "Greedy longest match over a vocabulary."},  # 10
    {"role": "user", "content": "Which one does BERT use?"},  # 9
]
```

There are three turns: messages 1–2, messages 3–4, and message 5. The three candidates cost $3 + 48 = 51$ tokens ($k = 0$), $51 - 17 = 34$ tokens ($k = 1$), and $34 - 16 = 18$ tokens ($k = 2$).

```python
trim_messages(messages, count_tokens, 67, 16)  # all six: 51 + 16 <= 67, exactly
trim_messages(messages, count_tokens, 64, 16)  # messages 0, 3, 4, 5: 34 + 16 <= 64
trim_messages(messages, count_tokens, 40, 16)  # messages 0, 5:       18 + 16 <= 40
trim_messages(messages, count_tokens, 33, 16)  # ValueError:          18 + 16 > 33
```

In the second call, the full conversation (51 tokens) is below `max_context = 64`. It is still trimmed because 16 tokens must be reserved for the answer.

## Constraints

- `messages` has between 1 and 200 messages, contains at least one `"user"` message, and follows the structure described above.
- `max_context` and `max_new_tokens` are integers with `1 <= max_context` and `0 <= max_new_tokens`. `max_new_tokens` may exceed `max_context`, in which case nothing fits.
- `count_tokens` is deterministic and has no side effects. You may call it as often as needed on lists built from the given messages.
- Results are compared for exact equality (the same messages in the same order).

## Why this matters

When a prompt and its requested output exceed the context window, a server has three choices. It can reject the request, let generation hit the limit and cut off mid-sentence, or trim the history. Chat front ends usually trim, and bugs in trimming fail quietly.

Forgetting to reserve `max_new_tokens` causes long conversations to end mid-sentence. Dropping the system message makes the assistant lose its instructions twenty turns in. Cutting inside a turn leaves an assistant reply without its question or a tool result without its tool call, which many chat templates reject outright. Summing per-message token counts misses template-level tokens around and between messages, so a prompt that appears to fit still overflows the window by a few tokens.
