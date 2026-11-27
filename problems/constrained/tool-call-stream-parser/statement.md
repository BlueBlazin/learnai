A model that calls tools writes the call into its ordinary text stream:
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. The server has to turn
that stream into the events of the OpenAI streaming format *while it arrives*, in
pieces that end wherever the tokenizer and the network put the boundaries: inside the
tag, inside the name, between a backslash and the quote it escapes. Write the
incremental parser. It must forward text as soon as it is certain to be text, never
forward something that may yet turn out to be a tag, and produce the same result
however the stream is cut.

![This problem is Parse tool calls, which turns the generated text into streaming events.](assets/map.svg)

## Specification

```python
def parse_start(): ...
def parse_step(state, piece): ...  # -> (state, events)
def parse_end(state): ...  # -> events
```

- **`parse_start()`** returns the state before any text has arrived.
- **`parse_step(state, piece)`** consumes `piece`, a `str` of any length (possibly
  empty), and returns the new state and the list of events that this piece makes
  certain.
- **`parse_end(state)`** is called once, when the stream is over, and returns the
  final events.

A state may be any value. **`parse_step` must not modify the state it is given**:
the caller may keep an old state and step it again.

### The text

The stream is ordinary text in which zero or more tool calls are embedded. A tool
call is, exactly:

```
<tool_call> ws { ws "name" ws : ws NAME ws , ws "arguments" ws : ws ARGS ws } ws </tool_call>
```

where `ws` is any amount of JSON whitespace (space, `\n`, `\r`, `\t`), `NAME` is a
JSON string, and `ARGS` is a JSON object. `"name"` always comes before
`"arguments"`. Everything outside tool calls is **content**. In content, only the
exact string `<tool_call>` begins a call; any other text, including `<`, `<tool`,
and `</tool_call>`, is content.

The tests feed streams of this form only, with one liberty: the stream may **end
early**, at any character.

### The events

Each event is a tuple:

| event | meaning |
|---|---|
| `("content", text)` | `text` is content, in order |
| `("tool_call_start", index, name)` | call number `index` has begun; `name` is its name |
| `("arguments", index, text)` | `text` is the next part of the arguments of call `index` |
| `("tool_call_end", index)` | call `index` is complete |

- `index` counts the calls of the stream from 0.
- `name` is the **value** of the `NAME` string: without its quotes and with its JSON
  escapes resolved (`"get_x"` gives `get_x`).
- The `text` parts of the `arguments` events of one call, concatenated, are `ARGS`
  **exactly as written**: every character from its opening `{` to the `}` that
  matches it, whitespace and escapes untouched, and nothing else. The `}` that
  closes the call object is not part of it. `ARGS` ends at its matching brace:
  braces, brackets, quotes, and tags *inside its strings* are data, and inside a
  string `\"` does not end the string while `\\` followed by `"` does.

### When events are emitted

Every event is emitted by the step whose piece delivers the character that makes it
certain:

1. **Content** is emitted as it arrives, with one exception. After each step, the
   longest suffix of the content received so far that is a proper prefix of
   `<tool_call>` is **held back**, because it may be the beginning of a tag. Held
   text is emitted by the first later step that shows it was not (or by
   `parse_end`), and is never emitted if it becomes a tag.
2. **`tool_call_start`** is emitted by the step that delivers the closing quote of
   `NAME`.
3. **`arguments`**: every character of `ARGS` is emitted by the step that delivers
   it.
4. **`tool_call_end`** is emitted by the step that delivers the final `>` of
   `</tool_call>`.

Within one step the events are in the order of the text. Consecutive `content`
events are merged into one, and so are consecutive `arguments` events of the same
call; no event has an empty `text`. A step that makes nothing certain returns `[]`.
This determines the event list of every step exactly.

**`parse_end(state)`** returns `[("content", held)]` if the stream ended in content
with held-back text, and `[]` otherwise. In particular a stream that ends inside a
tool call gets no `tool_call_end` and no error: the events already emitted stand.

You may use the `json` module (for example to decode `NAME`).

## Example

Seven pieces, then the end of the stream:

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

(`⏎` is a line feed. Event texts are written as Python literals, so `\\` is one
backslash.)

- Piece 1 ends with `<`, which could begin a tag, so only `Checking ` is emitted.
  Piece 2 begins with `b`: the `<` was content, and it leaves with the rest. The
  same piece ends with `<tool`, which is held and, this time, becomes a tag.
- The name is split across pieces 3 and 4. Nothing is emitted for a partial name.
- Piece 5 ends with a backslash inside a string. The quote that begins piece 6 is
  escaped by it and does not end the string, so the `}` after it is data. The
  arguments end at the first `}` of `}}`; the second closes the call.
- The arguments of call 0 concatenate to `{"city": "Oslo", "note": "a \"}\" b"}`,
  which is valid JSON.
- The stream ends with `<to` still held. It never became a tag, so `parse_end`
  releases it as content.

## Constraints

- Streams have at most a few thousand characters and may contain any Unicode
  characters; pieces may be empty.
- Event lists are compared exactly, step by step. The tests also cut the same text
  in many different ways and compare the results: after merging adjacent `content`
  events and adjacent `arguments` events of the same call across steps, every
  chunking must give the same list.
- Tuples, not lists; `index` is an `int`; `text` and `name` are `str`.

## Why this matters

This parser sits between the model and every agent framework that speaks the OpenAI
API. Its events map one to one onto streaming chunks: `content` becomes
`delta.content`, `tool_call_start` the first `delta.tool_calls` entry of a call
(with its `index`, an id, and `function.name`), and each `arguments` event a chunk
with `function.arguments`, which the client concatenates and parses when the call is
done. The failures are familiar from production. A tag split over two tokens leaks
into the chat window as `<tool` and the call is lost. Argument fragments that repeat
from the start concatenate into invalid JSON on the client. A second call that
reuses index 0 is appended to the first one's arguments. And a parser that waits for
the closing tag before emitting anything is correct and turns a streaming endpoint
into a blocking one.
