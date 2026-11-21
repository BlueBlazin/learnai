Build the server that stands between an application and a model when the application
needs more than text: an OpenAI-compatible `/v1/chat/completions` endpoint that
streams, that can force every response to be a document of a JSON Schema, that
reports tool calls in the wire format agents expect, and that neither generates for a
client who has left nor compiles the same schema twice. You have written each part
before: the endpoint, the schema-to-regex translation, the token index of an
automaton, the tool-call parser. The capstone is their composition, and the places
where they meet: the model now hands you **logits**, and the server chooses every
token.

## Overview

```python
def make_app(lm, vocab):
    async def app(scope, receive, send): ...

    return app
```

`make_app` returns an ASGI application; the tests serve it with uvicorn on a local
socket and talk to it over HTTP. The starter is written for the raw ASGI interface
and provides `read_body`, `send_json`, `sse_message`, `render_prompt`, and
`error_body`; Starlette may be used instead. `make_app` is the only name the tests
import. The starter also names the functions of the earlier problems that the stages
need (`schema_to_regex`, `build_index`, `parse_step`, ...): they are a suggested
layout, not an interface.

![The request is validated and its schema, if any, is looked up in the compilation cache or compiled once. The loop takes one row of logits per step from the model, masks it with the token index, and picks the best allowed token; the tool-call parser and the renderer turn the tokens into deltas, which are either streamed as chunks or folded into one message.](assets/architecture.svg)

The cases are grouped into four stages, and each stage's cases send only requests
that this stage and the earlier ones define: stage 1 never sends `response_format` or
`tools`, stage 2 never sends `tools`, and only stage 4 disconnects or counts
compilations. So the stages can be solved, and pass, in order.

**The model.** `lm` is a
[`ScriptedLM`](../../../python/src/learnai/kit/scriptedlm.py) whose script is made
of logits:

- `lm.stream(prompt, max_tokens)` returns an async iterator. Each item is one
  generation step: a **row of logits**, a list of `vocab.size` floats, where entry
  `t` is the model's score for token `t` as the next token. `max_tokens` is an int or
  `None` (no limit); the iterator yields at most that many rows. The rows are
  scripted: they do not depend on the tokens the server chose.
- `await stream.aclose()` ends the generation. The model does not know which token
  the server picked, so it never stops by itself before its rows run out: **the
  server closes the stream** as soon as it needs no further row.
- `lm.count_tokens(text)` is the number of tokens of a prompt.

Call `lm.stream` exactly once per valid request and never for an invalid one.

**The vocabulary.** `vocab` is a
[`ToyVocab`](../../../python/src/learnai/kit/vocab.py) with 282 tokens.
`vocab.bytes(t)` is the text of token `t` as `bytes`; every single byte is a token,
and so are a few longer pieces such as `{"`, `":`, `": `, `"}`, `, `, `true`, `tru`,
and `nul`. Ids 0 to 4 are special tokens with empty bytes; `vocab.eos_id` (1) is
end-of-sequence.

## Stage 1: chat completions

This is the endpoint of `openai-chat-endpoint` with one change: the server decodes.

**Routing**, in this order: a path other than exactly `/v1/chat/completions` gets
`404`; a method other than `POST` gets `405` with the header `allow: POST`; an invalid
body gets `400`; otherwise the response has status `200`.

**The request body** is a JSON object. Fields not named in this statement are
ignored.

| field | rule |
|---|---|
| `model` | required; a non-empty string |
| `messages` | required; a non-empty array of objects, each with `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (a string, possibly empty) |
| `max_completion_tokens` | optional; an integer `>= 1` |
| `max_tokens` | optional; an integer `>= 1`; used only if `max_completion_tokens` is absent |
| `stream` | optional; a boolean, default `false` |
| `stream_options` | optional; an object whose `include_usage`, if present, is a boolean |

For every optional field of this statement, `null` means absent. `true` and `false`
are not integers. With neither limit, the model is called with `max_tokens=None`.

**Errors.** Every `400` is `application/json` with the body

```json
{"error": {"message": "<text>", "type": "invalid_request_error", "param": "<field>", "code": null}}
```

`message` is a non-empty string of your choice; `param` is the name of the top-level
field that is invalid, or `null` when the body is not a JSON object at all. No test
request has more than one invalid field. For `404` and `405` only the status and an
`error` object with a `message` are checked.

**The prompt** is the conversation in ChatML, which `render_prompt` produces: for
each message `<|im_start|>{role}\n{content}<|im_end|>\n`, then
`<|im_start|>assistant\n`.

**Generation** is greedy. For each row the model yields, in order:

1. The **candidates** are all tokens (stage 2 narrows them).
2. The chosen token is the candidate with the largest logit; among equal logits, the
   one with the **lowest id**.
3. If it is end-of-sequence, the generation is over. Otherwise its text,
   `vocab.bytes(t).decode()`, is appended to the output, and the next row is read.

When the generation is over, by end-of-sequence or because the rows ran out, the
server closes the model's stream and reads no further row. Then, with $c$ the number
of rows read:

- `finish_reason` is `"stop"` if end-of-sequence was chosen, and `"length"`
  otherwise: rows that ran out mean the limit cut the generation off.
- `usage` is `{"prompt_tokens": lm.count_tokens(prompt), "completion_tokens": c,
  "total_tokens": <their sum>}`. End-of-sequence is a token the model spent a step
  on: it counts.

> [!NOTE]
> This differs from `openai-chat-endpoint`, whose model hid its stop token and where
> a generation of exactly `max_tokens` pieces counted as cut off. Here the stop token
> is visible: a generation whose last allowed row chooses end-of-sequence finished by
> itself, and its `finish_reason` is `"stop"`.

**Common fields.** Every response object carries `"id"`, a string that starts with
`chatcmpl-` and is different for every request; `"created"`, the Unix time in whole
seconds; and `"model"`, the request's `model`.

**Without `stream`** the response is one `application/json` object:

```json
{"id": "...", "object": "chat.completion", "created": 1794648729, "model": "...",
 "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "<the output>"},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 13, "total_tokens": 21}}
```

**With `"stream": true`** the response is `text/event-stream`: server-sent events
`data: <payload>\n\n`. All chunks of one response have the same `id`, `created`, and
`model`, and `"object": "chat.completion.chunk"`. In order:

1. **The role chunk**: `"choices": [{"index": 0, "delta": {"role": "assistant",
   "content": ""}, "finish_reason": null}]`.
2. **One chunk per chosen token** other than end-of-sequence: `"delta": {"content":
   "<its text>"}`, `"finish_reason": null`.
3. **The finish chunk**: `"delta": {}` and the `finish_reason`.
4. **The usage chunk**, only if `stream_options.include_usage` is `true`:
   `"choices": []` and `"usage": {...}`; every earlier chunk then carries `"usage":
   null`. Otherwise no chunk has a `usage` other than `null`.
5. **The terminator**: the literal `data: [DONE]`, then the end of the response.

Each chunk is sent in its own `http.response.body` message as soon as its token is
chosen, before the next row is asked for.

### Example 1

The model's rows put their largest logit on `H`, `e`, `l`, `l`, `o`, `, ` (one
token), `w`, `o`, `r`, `l`, `d`, `!`, then end-of-sequence, then `?` twice. For the
messages `system: "You are terse."`, `user: "Say hello"` (a prompt of 8 tokens), the
response is the JSON object above with the content `Hello, world!`: 13 rows are read,
the thirteenth chooses end-of-sequence, the stream is closed, and the two `?` rows are
never asked for.

With `"max_tokens": 4, "stream": true, "stream_options": {"include_usage": true}`
(the common fields are left out of this listing):

```
data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": "H"}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": "e"}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": "l"}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": "l"}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": null}

data: {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}

data: [DONE]

```

With `"max_tokens": 12` the content is complete and the `finish_reason` is still
`"length"`; with 13 it is `"stop"`.

## Stage 2: schema-constrained output

`response_format` (optional) selects what the content may be:

| value | meaning |
|---|---|
| absent, `null`, or `{"type": "text"}` | no constraint |
| `{"type": "json_schema", "json_schema": {"name": "<non-empty string>", "schema": {...}}}` | the content is a compact JSON document that conforms to `schema` |

Anything else (`{"type": "json_object"}`, a missing or empty `name`, a missing
`schema` or one that is not an object, a schema outside the subset below) is a `400`
with `param` `"response_format"`. Other keys of `json_schema` (`description`,
`strict`) are ignored.

**The pipeline** is the one of the three earlier problems:

1. **Schema to pattern.** A regular expression that matches exactly the compact JSON
   documents of the schema, by the rules of `json-schema-regex`, restated below. This
   part is yours.
2. **Pattern to automaton.** `dfa = CharDFA.from_regex(pattern)` from the kit: a
   deterministic automaton over bytes, minimal and trimmed, so every state it has can
   still reach an accepting state. **Use the kit's compiler for this step**; stage 4
   counts its calls. `dfa.start` is the start state, `dfa.states` all states,
   `dfa.step(state, byte)` the next state or `DEAD` (-1), `dfa.walk(state, data)`
   several steps, `dfa.is_accepting(state)` and `dfa.accepting` the accepting states.
3. **Automaton to token index.** Token `t` is **allowed** in state `q` when
   `vocab.bytes(t)` has at least one byte and reading all of its bytes from `q` never
   yields `DEAD`; the state reached is its next state. This part is yours as well,
   built through `dfa.step` and `dfa.walk` only: the kit's `DFAConstraint` and the
   automaton's test oracles (`accepts`, `accepts_prefix`, `is_live`) are rejected by
   the harness.

**Constrained generation** changes step 1 of the loop. With the automaton in state
`q` (the start state before the first row), the candidates are the tokens allowed in
`q`, plus end-of-sequence **exactly when `q` is accepting**. The choice among them is
as before (largest logit, lowest id), and a chosen token moves the automaton to its
next state. Everything else is unchanged: the response has the same shape, streamed
or not, `finish_reason` is `"stop"` only if end-of-sequence was chosen, and a
generation that runs out of rows inside a document reports `"length"` with the
unfinished document as its content.

So the content is always a prefix of a document of the schema, and with
`finish_reason: "stop"` it is a whole one, whatever the model's logits prefer.

**Schemas and their documents** (the subset of `json-schema-regex`). *Compact* means
no whitespace outside strings.

| schema | documents |
|---|---|
| `{"type": "null"}` | `null` |
| `{"type": "boolean"}` | `true` or `false` |
| `{"type": "integer"}` | an optional `-`, then `0` or a digit `1`–`9` followed by any digits |
| `{"type": "number"}` | an integer, then optionally `.` and one or more digits, then optionally `e` or `E`, an optional sign, and one or more digits |
| `{"type": "string"}` | `"`, string items, `"`. An item is one character other than `"`, `\`, and U+0000–U+001F; or `\` and one of `"\/bfnrt`; or `\u` and four hex digits. `minLength` and `maxLength` (optional) bound the number of items |
| `{"enum": [...]}` | exactly `json.dumps(v)` for one of the values (printable ASCII strings, ints, booleans, `null`); other keys are ignored |
| `{"type": "array", "items": S}` | `[`, documents of `S` separated by `,`, `]`; `minItems` (default 0) and `maxItems` (default: none) bound the count |
| `{"type": "object", "properties": {...}, "required": [...]}` | `{`, members `"name":value` separated by single commas, `}` |

In an object, members appear in the order in which `properties` declares them; a
required property appears exactly once and any other at most once; no other member is
allowed; `{}` has no comma. Property names are printable ASCII without `"` and `\`;
names and enum values are data and match only themselves. Digits are ASCII digits. A
schema without `enum` whose `type` is not one of the seven above (also when it is
nested) is unsupported: the `400` above. The pattern must stay inside what
`CharDFA.from_regex` compiles (literals, classes, `|`, groups, `* + ?`, `{m,n}` with
bounds up to 100; no anchors, lookaround, or lazy quantifiers).

### Example 2

```json
{"type": "object",
 "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}},
 "required": ["ok"]}
```

The model would like to write loosely formatted JSON with a quoted value. Each row
below ranks the listed tokens highest, in this order; every other token scores lower.

| row | the model's ranking | chosen | why |
|---|---|---|---|
| 1 | `{"` | `{"` | allowed: an object begins, then a name |
| 2, 3 | `o`; `k` | `o`, `k` | |
| 4 | `": ` then `":` | `":` | the token with a space is not compact |
| 5 | `"` then `true` | `true` | a boolean cannot begin with a quote |
| 6 | `, ` then `,` then `}` | `,` | the optional member `n` may follow |
| 7, 8 | `"`; `n` | `"`, `n` | |
| 9 | `": ` then `":` | `":` | |
| 10 | `0` | `0` | |
| 11 | `7` then `}` | `}` | no digit may follow a leading zero |
| 12 | a line feed, then end-of-sequence | end-of-sequence | the document is complete: nothing else is a candidate |

The content is `{"ok":true,"n":0}`, `completion_tokens` is 12, and `finish_reason` is
`"stop"`. The model's own top choices spell `{"ok": ", "n": 07` and a line feed.
With `"max_tokens": 5` the content is `{"ok":true` and the reason is `"length"`.

## Stage 3: tool calls

Two more optional fields:

| field | rule |
|---|---|
| `tools` | a non-empty array of `{"type": "function", "function": {"name": "<non-empty string>", "description": ..., "parameters": {...}}}`; `description` is ignored, `parameters` is optional and, if present, an object (a JSON Schema). Otherwise `400` with `param` `"tools"` |
| `tool_choice` | `"none"`, `"auto"`, or `{"type": "function", "function": {"name": "<name>"}}` naming one of `tools`. Default: `"auto"` if `tools` is present, else `"none"`. `"auto"` or a named choice without `tools`, an unknown name, and any other value (including `"required"`, which this server does not offer) are a `400` with `param` `"tool_choice"` |

The prompt does not change (a real chat template would list the tools). No test
combines `tools` with a `json_schema` response format.

**`"none"`**: the request is handled as in stages 1 and 2, whatever the model writes.

**A named choice** forces one call of that tool. The generation is constrained, as
in stage 2, by the tool's `parameters` schema (without `parameters`:
`{"type": "object", "properties": {}}`, whose only document is `{}`), and its whole
output is the arguments of the call. An unsupported `parameters` schema of the named
tool is a `400` with `param` `"tools"`; the schemas of other tools are not looked at.

**`"auto"`**: the generation is unconstrained, and the model may write tool calls
into its text as

```
<tool_call> ws { ws "name" ws : ws NAME ws , ws "arguments" ws : ws ARGS ws } ws </tool_call>
```

where `learnai` is any amount of JSON whitespace, `NAME` a JSON string, and `ARGS` a JSON
object. The server parses the output incrementally, token by token, by the rules of
`tool-call-stream-parser`:

- Everything outside tool calls is content. After each token, the longest suffix of
  the content so far that is a proper prefix of `<tool_call>` is **held back**; it is
  released by the first later token that shows it was not a tag (or at the end of the
  generation), and never if it becomes a tag. In content, only the exact string
  `<tool_call>` begins a call.
- A call **starts** at the closing quote of `NAME`; its name is the value of that
  JSON string (escapes resolved). Calls are numbered from 0 in order.
- Every character of `ARGS`, from its opening `{` to the matching `}`, exactly as
  written, is an **argument fragment** of the call, available with the token that
  delivers it. Braces, quotes, and tags inside the strings of `ARGS` are data.
- The call ends with the `>` of `</tool_call>`; text after it is content again. The
  model writes well-formed calls, but the generation may end anywhere.
- Within one token, adjacent content is one piece and adjacent argument fragments of
  one call are one piece.

The server does not check that a called name is one of `tools`.

**Deltas.** In a stream, after the role chunk, each of the following is one chunk
with `"finish_reason": null`, in the order of the text:

| what | `delta` |
|---|---|
| a piece of content | `{"content": "<text>"}` |
| the start of call `i` | `{"tool_calls": [{"index": i, "id": "<id>", "type": "function", "function": {"name": "<name>", "arguments": ""}}]}` |
| an argument fragment of call `i` | `{"tool_calls": [{"index": i, "function": {"arguments": "<text>"}}]}` |

A token that makes nothing certain produces no chunk; content released at the end of
the generation is one last content chunk before the finish chunk. `id` is a string
that starts with `call_` and is different for every call; only the start delta
carries `id`, `type`, and `name`. With a named choice, the start delta of call 0
follows the role chunk at once, before the first row is read, and every chosen token
is one argument fragment.

**Without `stream`**, the message is what the deltas add up to: `content` is the
concatenated content, and `"tool_calls": [{"id": ..., "type": "function", "function":
{"name": ..., "arguments": "<the concatenated fragments>"}}, ...]` lists the calls in
order. If no call started, the message has no `tool_calls` key. If at least one call
started and the content is empty, `content` is `null`.

**`finish_reason`**: `"length"` if end-of-sequence was not chosen; otherwise
`"tool_calls"` if at least one call started; otherwise `"stop"`. Usage is as before.

### Example 3

`tools` holds `get_weather` and `get_time`; `tool_choice` is absent. The model's top
choices spell, in 76 tokens,

```
Checking.<tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>
```

and then end-of-sequence. Streamed, between the role chunk and the finish chunk:

| tokens | chunks |
|---|---|
| 1 to 9: `C` ... `.` | nine content chunks, one per token |
| 10 to 38: `<` ... `r` | none: the tag is held back, and the name is not complete |
| 39: `"` | the start of call 0: `id`, `"type": "function"`, `"name": "get_weather"`, `"arguments": ""` |
| 40 to 51 | none |
| 52 to 63: `{"`, `c`, `i`, `t`, `y`, `": `, `"`, `O`, `s`, `l`, `o`, `"}` | twelve argument fragments, one per token |
| 64 to 76: `}`, `<` ... `>` | none |

The finish chunk has `"finish_reason": "tool_calls"`, and `completion_tokens` is 77.
Not streamed, the message is

```json
{"role": "assistant", "content": "Checking.",
 "tool_calls": [{"id": "call_6f1c...", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]}
```

With `"max_tokens": 22` the output ends inside the name: the content is `Checking.`,
no call has started, and the reason is `"length"`. With `"tool_choice": "none"`, or
without `tools`, the whole text is content and the reason is `"stop"`.

## Stage 4: cancellation and the compilation cache

**Client disconnect.** If the client closes the connection during a stream, the
server stops reading rows and closes the model's stream. As in `sse-token-stream`,
`send` does not fail on a closed connection with this server; the disconnect is
announced by `receive()` returning `{"type": "http.disconnect"}`. Requests that are
not streamed are not disconnected in the tests.

**Compile each schema once.** Compiling a schema is the expensive part of a
constrained request, and applications send the same few schemas over and over. Each
app (each call of `make_app`) keeps a cache of compiled schemas:

- Two schemas are **the same** when their JSON texts are equal with the keys in the
  order they were sent: `json.dumps(schema, separators=(",", ":"))` of the parsed
  schema, or any hash of it. Whitespace in the request does not matter. The order of
  keys does: `properties` declared in another order is another language.
- The first request that needs a schema calls `CharDFA.from_regex` **exactly once**
  for it and builds its **whole** token index, every state of the automaton, before
  answering.
- Every later request that needs the same schema calls `CharDFA.from_regex` **zero**
  times and makes **zero** calls of `dfa.step` or `dfa.walk`, whatever its tokens
  are. The schema of a `response_format` and the `parameters` of a named tool share
  the cache: it is keyed by the schema, not by where it came from or by the
  `json_schema.name`.

## Constraints

- Request bodies are at most 64 KiB; a generation has at most 1,000 rows; a schema's
  automaton has at most a few hundred states. Compiling one of the test schemas and
  indexing 282 tokens takes well under a second.
- **Inputs you may rely on.** Rows have exactly `vocab.size` finite floats. A chosen
  token is never a special token other than end-of-sequence, and its bytes are always
  valid UTF-8 by themselves (the tests' rows score the tokens that hold only part
  of a character far below every other token). In every state the tests reach, at least one token is a
  candidate.
- Responses are compared after JSON decoding (key order and spacing are free; extra
  fields such as `"logprobs": null` are allowed) with what the rules above give for
  the model's rows: the content, the tool calls, the chunks one by one, the finish
  reason, the usage, and the number of rows read from the model. Ids are checked for
  their prefix and for being distinct.
- Complete constrained documents are also checked against the schema directly.
- Incremental delivery is checked by order, not by timing: the model is gated and
  yields row `i + 1` only after the test has received the chunk of token `i`.
- For the disconnect checks the model yields a row every 5 ms; the client reads a few
  chunks and closes the connection. The case fails if the model is drained to its end
  or its stream is not closed within 5 s.
- **Stage 4 counts** `CharDFA.from_regex` calls and the `dfa.step` calls of every
  automaton compiled while the case runs, per request. There is no tolerance: one
  compilation for a new schema, none for a known one, and no step for a known one.
  The schemas of these cases are not used anywhere else, so a cache finds them
  missing the first time.
- Do not block the event loop (no `time.sleep`, no synchronous I/O).

## Why this matters

Structured output is where serving and decoding stop being separate concerns. An
agent framework sends a schema with every call and reads the reply with a JSON
parser; a server that only asks the model nicely returns prose, or valid JSON with a
trailing comma, a few times in a thousand, and the failure surfaces as a crashed
tool three services away. Masking tokens makes the guarantee unconditional, and
moves the problems to the seams this capstone is about. A constraint applied on one
response path and forgotten on the other passes every non-streaming test. A document
cut off by `max_tokens` is not a document, and only `finish_reason` says so. A tool
call whose id or name is repeated in every fragment is opened again and again by the
client. A schema compiled per request adds its compile time to every first token,
and a cache keyed by the wrong notion of "same" either never hits or, worse, answers
with the automaton of a different schema. And a generation that continues for a
client who has gone holds a slot that a waiting request needs.
