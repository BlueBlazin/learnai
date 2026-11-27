A model that invokes tools writes each call directly into its text stream as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. Implement an incremental parser (`parse_start`, `parse_step`, `parse_end`) that converts this stream into OpenAI-style streaming events as arbitrary text pieces arrive. The parser must emit content as soon as it cannot be part of a tag, hold back any suffix that might start a tag, and produce the same merged output regardless of how the stream is split.

![This problem is Parse tool calls, which turns the generated text into streaming events.](assets/map.svg)

## Specification

```python
def parse_start(): ...
def parse_step(state, piece): ...  # -> (state, events)
def parse_end(state): ...  # -> events
```

- **`parse_start()`** returns the initial state before any text has arrived.
- **`parse_step(state, piece)`** consumes `piece`, a `str` of any length (including `""`), and returns `(new_state, events)` where `events` is the list of events made certain by `piece`.
- **`parse_end(state)`** is called once at the end of the stream and returns any final events.

A state may be any value. **`parse_step` must not modify the state it receives**, because the caller may keep an earlier state and step from it again.

### The text

The stream is ordinary text with zero or more embedded tool calls. Every tool call has the exact form:

```
<tool_call> ws { ws "name" ws : ws NAME ws , ws "arguments" ws : ws ARGS ws } ws </tool_call>
```

Here `ws` is any amount of JSON whitespace (space, `\n`, `\r`, `\t`), `NAME` is a JSON string, and `ARGS` is a JSON object. `"name"` always appears before `"arguments"`. Everything outside tool calls is **content**. In content, only the exact string `<tool_call>` starts a call. Any other text, including `<`, `<tool`, and `</tool_call>`, is content.

The tests feed streams of this form only, except that a stream may **end early** at any character.

### The events

Each event is a tuple:

| event | meaning |
|---|---|
| `("content", text)` | `text` is content, in order |
| `("tool_call_start", index, name)` | call number `index` has begun with decoded tool name `name` |
| `("arguments", index, text)` | `text` is the next fragment of the arguments for call `index` |
| `("tool_call_end", index)` | call `index` is complete |

- `index` counts tool calls in the stream starting from `0`.
- `name` is the decoded **value** of the `NAME` JSON string, without its surrounding quotes and with JSON escapes resolved (for example, `"get_x"` becomes `get_x`).
- Concatenating the `text` fields of all `arguments` events for a call produces `ARGS` **verbatim**, from its opening `{` through its matching `}`, with whitespace and escapes untouched. The `}` that closes the outer tool-call object is not included.
- `ARGS` ends at its matching closing brace. Braces, brackets, quotes, and tags inside its JSON strings are ordinary data. Inside a string, `\"` does not end the string, while `\\` followed by `"` does.

### When events are emitted

Every event is emitted by the `parse_step` call that receives the character making that event certain.

1. **Content** is emitted as it arrives, with one exception. After each step, the longest suffix of the current content that is a proper prefix of `<tool_call>` is **held back** because it might begin a tag. Held text is emitted by the first later step that proves it is not a tag (or by `parse_end`), and is never emitted if it completes `<tool_call>`.
2. **`tool_call_start`** is emitted by the step that delivers the closing quote of `NAME`.
3. **`arguments`** events emit every character of `ARGS` during the step that delivers it.
4. **`tool_call_end`** is emitted by the step that delivers the final `>` of `</tool_call>`.

Within a single step, events appear in the order of the text. Merge consecutive `content` events into one, and merge consecutive `arguments` events of the same call into one. No event may have an empty `text`. A step that makes nothing certain returns `[]`. These rules determine the event list of every step uniquely.

**`parse_end(state)`** returns `[("content", held)]` if the stream ended in content with held-back text, and `[]` otherwise. If a stream ends inside a tool call, `parse_end` returns `[]` with no error and no `tool_call_end`, leaving the events already emitted intact.

You may use the `json` module (for example, to decode `NAME`).

## Example

Seven pieces arrive, followed by the end of the stream:

| piece | events |
|---|---|
| `Checking <` | `("content", "Checking ")` |
| `b>now</b>.<tool` | `("content", "<b>now</b>.")` |
| `_call>{"name": "get_wea` | (none) |
| `ther", "arguments": {"city": "Os` | `("tool_call_start", 0, "get_weather")`, `("arguments", 0, '{"city": "Os')` |
| `lo", "note": "a \` | `("arguments", 0, 'lo", "note": "a \\')` |
| `"}\" b"}}</tool_call` | `("arguments", 0, '"}\\" b"}')` |
| `>⏎<tool_call>{"name": "noop", "arguments": {}}</tool_call> Done <to` | `("tool_call_end", 0)`, `("content", "\n")`, `("tool_call_start", 1, "noop")`, `("arguments", 1, "{}")`, `("tool_call_end", 1)`, `("content", " Done ")` |
| (end) | `("content", "<to")` |

Here `⏎` represents a newline (`\n`). Event texts are shown as Python string literals, so `\\` is a single backslash.

- Piece 1 ends with `<`, which could start a tag, so only `Checking ` is emitted. Piece 2 begins with `b`, proving `<` was content and releasing it with the rest. Piece 2 ends with `<tool`, which is held and becomes a tag in piece 3.
- The tool name is split across pieces 3 and 4. Nothing is emitted for a partial name.
- Piece 5 ends with a backslash inside a string. The leading quote of piece 6 is escaped by that backslash and does not close the string, so the `}` after it is data. The arguments end at the first `}` of `}}`, while the second closes the outer call object.
- The argument fragments of call 0 concatenate to `{"city": "Oslo", "note": "a \"}\" b"}`, which is valid JSON.
- The stream ends with `<to` still held. Because it never became a tag, `parse_end` releases it as content.

## Constraints

- Streams have at most a few thousand characters and may contain any Unicode characters. Pieces may be empty.
- Event lists are compared exactly, step by step. The tests also split the same stream in many different ways and verify that merging adjacent `content` events and adjacent `arguments` events of the same call across steps yields the same result for every chunking.
- Events must be tuples, not lists. `index` must be an `int`, and `text` and `name` must be `str`.

## Why this matters

This parser sits between the model and every agent framework that speaks the OpenAI streaming API. Its events map directly onto streaming chunks. A `content` event becomes `delta.content`, a `tool_call_start` event creates the first `delta.tool_calls` entry (with `index`, an ID, and `function.name`), and each `arguments` event sends a `function.arguments` fragment that the client concatenates and parses once the call finishes.

Small bugs here break production clients quietly. A tag split across two tokens leaks `<tool` into the chat window and drops the tool call. Cumulative argument fragments instead of deltas concatenate into invalid JSON on the client. Reusing index `0` for a second call appends its arguments onto the first call. Waiting for `</tool_call>` before emitting anything preserves correctness but turns a streaming endpoint into a blocking one.
