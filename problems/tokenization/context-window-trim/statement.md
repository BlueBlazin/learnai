A chat grows by two messages per exchange, and the context window does not. Before
every request, a chat server decides which part of the history still goes into the
prompt. Implement the standard policy: always keep the system message and the latest
turn, drop the oldest turns whole until the prompt fits, and leave room for the
answer. The work is in the bookkeeping: what a turn is, what "fits" means, and what
must never be dropped.

## Specification

```python
def trim_messages(
    messages: list[dict[str, str]],
    count_tokens: Callable[[list[dict[str, str]]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[dict[str, str]]: ...
```

- `messages`: the conversation, oldest first. Each message is a dict with the keys
  `"role"` (`"system"`, `"user"`, `"assistant"`, or `"tool"`) and `"content"`.
- `count_tokens(msgs)`: the number of tokens of the prompt that the model's chat
  template renders from the list `msgs`, including everything the template adds
  (special tokens, role headers, the generation prompt). It accepts any list of
  messages taken from `messages`. Treat it as a black box: the count of a list is
  **not** the sum of the counts of its messages.
- `max_context`: the model's context window, in tokens.
- `max_new_tokens`: the number of tokens reserved for the model's answer.

**Structure of a conversation.** If there is a system message, it is `messages[0]`,
and there is only one. The rest of the conversation is a sequence of **turns**. A turn
starts at a `user` message and contains every message up to, but not including, the
next `user` message: the user's message, the assistant's reply, and any tool calls and
tool results in between. The first message after the system message is always a
`user` message, so every message belongs to exactly one turn. The **last turn** starts
at the latest `user` message; it may consist of that message alone, or continue with
assistant and tool messages when the model is called again in the middle of a tool
loop.

**Fitting.** A list of messages `kept` fits when

$$
\texttt{count\_tokens(kept)} + \texttt{max\_new\_tokens} \le \texttt{max\_context} .
$$

**The result.** For $k = 0, 1, \dots, T - 1$, where $T$ is the number of turns, let
candidate $k$ be the system message (if there is one) followed by all turns except the
$k$ oldest, in their original order. Return candidate $k$ for the **smallest** $k$
that fits. So:

- If the whole conversation fits, it is returned unchanged ($k = 0$).
- Turns are dropped whole and oldest first. A dropped turn takes all its messages
  with it; the result never starts, after the system message, with an assistant or
  tool message.
- The system message and the last turn are never dropped. If even they do not fit
  (candidate $T - 1$), raise `ValueError`.

The function returns a new list holding the kept messages in order (the same dicts, or
equal ones). It does not modify `messages` or any message, also when it raises.

## Example

In this example a prompt costs 3 tokens plus, per message, 4 tokens and one token per
word of its content:

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

There are three turns: messages 1-2, messages 3-4, and message 5. The candidates cost
$3 + 48 = 51$ tokens ($k = 0$), $51 - 17 = 34$ ($k = 1$), and $34 - 16 = 18$ ($k = 2$).

```python
trim_messages(messages, count_tokens, 67, 16)  # all six: 51 + 16 <= 67, exactly
trim_messages(messages, count_tokens, 64, 16)  # messages 0, 3, 4, 5: 34 + 16 <= 64
trim_messages(messages, count_tokens, 40, 16)  # messages 0, 5:       18 + 16 <= 40
trim_messages(messages, count_tokens, 33, 16)  # ValueError:          18 + 16 > 33
```

In the second call the whole conversation, 51 tokens, is below `max_context = 64`. It
is still trimmed, because 16 tokens of the window are reserved for the answer.

## Constraints

- `messages` has between 1 and 200 messages and at least one `user` message, with the
  structure described above.
- `0 <= max_new_tokens` and `1 <= max_context`; both are ints. `max_new_tokens` may
  exceed `max_context`, in which case nothing fits.
- `count_tokens` is deterministic and has no side effects. You may call it as often as
  you like, on lists built from the given messages only.
- Results are compared exactly: the same messages, in the same order.

## Why this matters

When the prompt and the requested output do not fit, a server has three options:
reject the request, let generation run into the end of the window and stop early with
an answer cut off in the middle, or trim. Trimming is what chat front ends do, and its
mistakes are quiet. Forget to reserve the output and long conversations end
mid-sentence. Drop the system message and the assistant loses its instructions twenty
turns in. Cut through a turn and the prompt starts with an answer to a question the
model cannot see, or with a tool result whose call is gone, which many chat templates
reject outright. Counting per message and adding misses what the template inserts
around and between messages, so the "fitting" prompt is a few tokens too long.
