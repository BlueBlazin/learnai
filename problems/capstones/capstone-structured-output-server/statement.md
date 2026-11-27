Implement `make_app`, an OpenAI-compatible `/v1/chat/completions` ASGI server that decodes tokens from model logits, supports streaming and non-streaming responses, enforces JSON Schema constraints via a compiled DFA token index, parses and forces tool calls, cancels generation on client disconnect, and caches compiled schemas.

## Overview

```python
def make_app(lm, vocab):
    async def app(scope, receive, send): ...

    return app
```

`make_app` returns an ASGI application that the test suite serves with uvicorn on a local socket and calls over HTTP. The starter code uses the raw ASGI interface and provides `read_body`, `send_json`, `sse_message`, `render_prompt`, and `error_body`, though you may use Starlette instead. `make_app` is the only symbol the tests import. The other helper names in the starter (`schema_to_regex`, `build_index`, `parse_step`, and so on) are a suggested layout rather than a required interface.

![The path of one request. The token index masks each row of logits inside the loop, and the parser turns the chosen tokens into deltas.](assets/architecture.svg)

The test cases are grouped into four stages, and each stage sends only requests defined by that stage and earlier ones. Stage 1 never sends `response_format` or `tools`, stage 2 never sends `tools`, and only stage 4 disconnects clients or counts schema compilations. You can implement and test the stages in order.

**The model.** `lm` is a [`ScriptedLM`](../../../python/src/learnai/kit/scriptedlm.py) that yields rows of logits:

- `lm.stream(prompt, max_tokens)` returns an async iterator. Each yielded item is one generation step containing a list of `vocab.size` floats, where entry `t` is the model's score for token `t` as the next token. `max_tokens` is an int or `None` (no limit), and the iterator yields at most that many rows. The rows are scripted and do not depend on which tokens the server picks.
- `await stream.aclose()` ends the generation. Because the model does not see the chosen tokens, it never stops early on its own. **The server must close the stream** as soon as it needs no further rows.
- `lm.count_tokens(text)` returns the token count of `text`.

Call `lm.stream` exactly once per valid request and never for an invalid request.

**The vocabulary.** `vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py) with 282 tokens. `vocab.bytes(t)` returns the bytes for token `t`. Every single byte is a token, along with a few multi-byte pieces such as `{"`, `":`, `": `, `"}`, `, `, `true`, `tru`, and `nul`. Token ids 0 through 4 are special tokens with empty bytes, and `vocab.eos_id` (1) is end-of-sequence.

## Stage 1: chat completions

This stage implements the `openai-chat-endpoint` API, except the server now decodes tokens directly from logits.

**Routing.** Check conditions in this exact order:
1. Any path other than `/v1/chat/completions` returns `404`.
2. Any method other than `POST` returns `405` with header `allow: POST`.
3. An invalid request body returns `400`.
4. Otherwise, return status `200`.

**Request body.** The body must be a JSON object. Ignore fields not listed in this statement.

| field | rule |
|---|---|
| `model` | required; a non-empty string |
| `messages` | required; a non-empty array of objects, each with `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (a string, possibly empty) |
| `max_completion_tokens` | optional; an integer `>= 1` |
| `max_tokens` | optional; an integer `>= 1`; used only if `max_completion_tokens` is absent |
| `stream` | optional; a boolean, default `false` |
| `stream_options` | optional; an object whose `include_usage`, if present, is a boolean |

For every optional field in this statement, `null` means absent. Booleans (`true` and `false`) are not valid integers. If neither token limit is present, call `lm.stream` with `max_tokens=None`.

**Errors.** Every `400` response has content type `application/json` and body:

```json
{"error": {"message": "<text>", "type": "invalid_request_error", "param": "<field>", "code": null}}
```

`message` is any non-empty string. `param` is the name of the invalid top-level field, or `null` if the body is not a JSON object. No test request contains more than one invalid field. For `404` and `405`, the tests check only the status code and the presence of an `error` object containing a `message`.

**Prompt.** Format the conversation in ChatML using `render_prompt`: for each message emit `<|im_start|>{role}\n{content}<|im_end|>\n`, followed by `<|im_start|>assistant\n`.

**Generation.** Decoding is greedy. For each row of logits yielded by the model, in order:

1. Build the **candidates** set (all tokens in stage 1; narrowed by the constraint in stage 2).
2. Choose the candidate with the largest logit, breaking ties toward the **lowest token id**.
3. If the chosen token is `vocab.eos_id`, stop generating. Otherwise, append `vocab.bytes(t).decode()` to the output and read the next row.

Once generation ends (either by choosing end-of-sequence or because the iterator runs out of rows), close the model stream and read no more rows. Let $c$ be the number of rows read:

- `finish_reason` is `"stop"` if end-of-sequence was chosen, and `"length"` otherwise (meaning the token limit cut off generation).
- `usage` is `{"prompt_tokens": lm.count_tokens(prompt), "completion_tokens": c, "total_tokens": prompt_tokens + c}`. End-of-sequence counts toward `completion_tokens` because the model spent a step producing it.

> [!NOTE]
> This differs from `openai-chat-endpoint`, where the model hid its stop token and a generation of `max_tokens` pieces counted as cut off. Here the stop token is explicit, so a generation whose final allowed row selects end-of-sequence finishes with `"stop"`.

**Common fields.** Every response object includes `"id"` (a string starting with `chatcmpl-` that is unique per request), `"created"` (Unix timestamp in whole seconds), and `"model"` (the request's `model`).

**Non-streaming response.** When `stream` is `false` or absent, return one `application/json` object:

```json
{"id": "...", "object": "chat.completion", "created": 1794648729, "model": "...",
 "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "<the output>"},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 13, "total_tokens": 21}}
```

**Streaming response.** When `"stream": true`, return `text/event-stream` formatted as server-sent events `data: <payload>\n\n`. Every chunk in a response shares the same `id`, `created`, and `model`, with `"object": "chat.completion.chunk"`. Send these events in order:

1. **Role chunk:** `"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]`.
2. **Content chunks:** one chunk per chosen non-EOS token with `"delta": {"content": "<its text>"}` and `"finish_reason": null`.
3. **Finish chunk:** `"delta": {}` and the `finish_reason`.
4. **Usage chunk (only if `stream_options.include_usage` is `true`):** `"choices": []` and `"usage": {...}`. When `include_usage` is `true`, every earlier chunk must include `"usage": null`. Otherwise, no chunk may have a non-null `usage`.
5. **Terminator:** the literal `data: [DONE]`, followed by the end of the response.

Send each chunk in its own `http.response.body` message immediately after choosing its token, before requesting the next row from `lm.stream`.

### Example 1

Suppose the model's rows rank `H`, `e`, `l`, `l`, `o`, `, ` (one token), `w`, `o`, `r`, `l`, `d`, `!`, end-of-sequence, `?`, `?` highest in order. For the messages `system: "You are terse."` and `user: "Say hello"` (an 8-token prompt), the non-streaming response is the JSON object above with content `Hello, world!`. The server reads 13 rows, stops on end-of-sequence at row 13, closes the stream, and never reads the two `?` rows.

With `"max_tokens": 4, "stream": true, "stream_options": {"include_usage": true}` (omitting common fields for brevity):

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

With `"max_tokens": 12`, the full text `Hello, world!` is emitted, but `finish_reason` is still `"length"`. With `"max_tokens": 13`, the thirteenth row chooses end-of-sequence and `finish_reason` is `"stop"`.

## Stage 2: schema-constrained output

The optional `response_format` field controls output constraints:

| value | meaning |
|---|---|
| absent, `null`, or `{"type": "text"}` | no constraint |
| `{"type": "json_schema", "json_schema": {"name": "<non-empty string>", "schema": {...}}}` | output must be a compact JSON document conforming to `schema` |

Any other value (`{"type": "json_object"}`, a missing or empty `name`, a missing or non-object `schema`, or an unsupported schema outside the subset below) returns `400` with `param` `"response_format"`. Ignore other keys inside `json_schema` such as `description` and `strict`.

**Compilation pipeline.** Combine the three steps from the prerequisite problems:

1. **Schema to regex.** Translate `schema` into a regular expression matching exactly its compact JSON documents, following the rules of `json-schema-regex` restated below. You implement this step.
2. **Regex to DFA.** Compile the pattern with `dfa = CharDFA.from_regex(pattern)` from the kit. This returns a minimal, trimmed byte-level DFA where every state can still reach an accepting state. **You must use the kit's compiler for this step** because stage 4 counts its calls. The DFA exposes `dfa.start`, `dfa.states`, `dfa.step(state, byte)` (returning the next state or `DEAD`, which is `-1`), `dfa.walk(state, data)`, `dfa.is_accepting(state)`, and `dfa.accepting`.
3. **DFA to token index.** Token `t` is **allowed** in state `q` if `vocab.bytes(t)` is non-empty and walking all of its bytes from `q` never reaches `DEAD`. The resulting state is its next state. You implement this step using only `dfa.step` and `dfa.walk`. The harness rejects any use of `DFAConstraint` or the DFA test oracles (`accepts`, `accepts_prefix`, `is_live`).

**Constrained generation.** Start in `q = dfa.start` before reading the first row. In step 1 of the generation loop, the candidates are the tokens allowed in `q`, plus `vocab.eos_id` **if and only if `q` is accepting**. Select the candidate with the largest logit (lowest id on ties). If a non-EOS token is chosen, advance `q` to its next state. Everything else remains identical. Response shapes are unchanged, `finish_reason` is `"stop"` only when end-of-sequence is chosen, and running out of rows mid-document returns `"length"` with the partial document as content.

As a result, the output is always a valid prefix of a conforming document, and whenever `finish_reason` is `"stop"` it is a complete conforming document regardless of the model's preferences.

**Supported JSON Schema subset** (same as `json-schema-regex`). *Compact* JSON has no whitespace outside strings.

| schema | documents |
|---|---|
| `{"type": "null"}` | `null` |
| `{"type": "boolean"}` | `true` or `false` |
| `{"type": "integer"}` | optional `-`, then `0` or a digit `1`–`9` followed by any ASCII digits |
| `{"type": "number"}` | an integer, optionally `.` and one or more digits, optionally `e` or `E` with an optional sign and one or more digits |
| `{"type": "string"}` | `"`, string items, `"`. An item is any character other than `"`, `\`, and U+0000–U+001F; or `\` followed by one of `"\/bfnrt`; or `\u` followed by four hex digits. Optional `minLength` and `maxLength` bound the number of items |
| `{"enum": [...]}` | exact `json.dumps(v)` for one of the values (printable ASCII strings, ints, booleans, `null`); ignore other keys |
| `{"type": "array", "items": S}` | `[`, documents of `S` separated by `,`, `]`; `minItems` (default 0) and `maxItems` (default: unbounded) bound the count |
| `{"type": "object", "properties": {...}, "required": [...]}` | `{`, members `"name":value` separated by single commas, `}` |

Object members must appear in the order declared in `properties`. Required properties appear exactly once, optional properties appear at most once, extra properties are forbidden, and `{}` has no comma. Property names are printable ASCII without `"` or `\`. Property names and enum values match only themselves as literals. Digits are ASCII digits. Any schema (top-level or nested) without `enum` whose `type` is not one of the seven types above is unsupported and returns `400`. The generated regex must stay within the syntax supported by `CharDFA.from_regex` (literals, character classes, `|`, groups, `*`, `+`, `?`, and `{m,n}` with bounds up to 100; no anchors, lookaround, or lazy quantifiers).

### Example 2

```json
{"type": "object",
 "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}},
 "required": ["ok"]}
```

Suppose the model prefers loosely formatted JSON with a quoted boolean. Each row below lists the highest-scoring tokens in descending order, with all other tokens scoring lower:

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
| 12 | a line feed, then end-of-sequence | end-of-sequence | the document is complete, so no other token is a candidate |

The response content is `{"ok":true,"n":0}`, `completion_tokens` is 12, and `finish_reason` is `"stop"`. Unconstrained, the model's top choices would have spelled `{"ok": ", "n": 07\n`. With `"max_tokens": 5`, the content is `{"ok":true` and `finish_reason` is `"length"`.

## Stage 3: tool calls

Stage 3 adds two optional request fields:

| field | rule |
|---|---|
| `tools` | a non-empty array of `{"type": "function", "function": {"name": "<non-empty string>", "description": ..., "parameters": {...}}}`. Ignore `description`. `parameters` is optional and, if present, must be an object (a JSON Schema). Otherwise return `400` with `param` `"tools"` |
| `tool_choice` | `"none"`, `"auto"`, or `{"type": "function", "function": {"name": "<name>"}}` naming one of `tools`. Defaults to `"auto"` when `tools` is present, otherwise `"none"`. Using `"auto"` or a named choice without `tools`, naming an unknown tool, or passing any other value (including `"required"`) returns `400` with `param` `"tool_choice"` |

The prompt stays unchanged (a production chat template would inject tool definitions). No test combines `tools` with a `json_schema` response format.

- **`"none"`.** Handle the request exactly as in stages 1 and 2, ignoring any tool markup the model writes.
- **Named choice.** Force a single call to the named tool. Constrain generation using the tool's `parameters` schema exactly as in stage 2 (if `parameters` is absent, use `{"type": "object", "properties": {}}`, which matches only `{}`). The entire generated output forms the tool call's arguments. If the named tool's `parameters` schema is unsupported, return `400` with `param` `"tools"`. Do not inspect the schemas of other tools.
- **`"auto"`.** Run unconstrained generation. The model may embed tool calls in its output using the format:

```
<tool_call> ws { ws "name" ws : ws NAME ws , ws "arguments" ws : ws ARGS ws } ws </tool_call>
```

Here `ws` is any JSON whitespace, `NAME` is a JSON string, and `ARGS` is a JSON object. Parse the output incrementally, token by token, following `tool-call-stream-parser`:

- Everything outside tool calls is regular content. After each token, **hold back** the longest suffix of the current content that is a proper prefix of `<tool_call>`. Release held-back text on the first later token that rules out `<tool_call>`, or when generation ends. Never release text that completes `<tool_call>`. Only the exact string `<tool_call>` starts a call.
- A call **starts** at the closing quote of `NAME`, with its name equal to the unescaped JSON string value. Number calls starting from 0 in order.
- Every character of `ARGS`, from its opening `{` through its matching `}` verbatim, is an **argument fragment** of the call and is emitted with the token that delivers it. Braces, quotes, and tags inside strings within `ARGS` are ordinary data.
- The call ends at the `>` of `</tool_call>`, after which text is content again. Tool calls are always well-formed, though generation may stop at any point.
- Within a single token, merge adjacent content into one piece and adjacent argument fragments of the same call into one piece.

Do not reject tool names that are absent from `tools`.

**Streaming deltas.** After the role chunk, emit one chunk with `"finish_reason": null` for each event in text order:

| event | `delta` |
|---|---|
| content piece | `{"content": "<text>"}` |
| start of call `i` | `{"tool_calls": [{"index": i, "id": "<id>", "type": "function", "function": {"name": "<name>", "arguments": ""}}]}` |
| argument fragment of call `i` | `{"tool_calls": [{"index": i, "function": {"arguments": "<text>"}}]}` |

A token that produces no confirmed output emits no chunk. Any content released when generation ends is emitted as a final content chunk before the finish chunk. Each call `id` starts with `call_` and is unique across calls. Only the start delta includes `id`, `type`, and `name`. For a named choice, emit the start delta for call 0 immediately after the role chunk (before reading the first row), and emit each chosen token as an argument fragment.

**Non-streaming response.** Fold the deltas into one assistant message. Set `content` to the concatenated content pieces, and set `"tool_calls"` to `[{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<concatenated fragments>"}}, ...]` in call order. Omit the `tool_calls` key if no call started. If at least one call started and the content is empty, set `content` to `null`.

**`finish_reason`.** Return `"length"` if end-of-sequence was not chosen. Otherwise, return `"tool_calls"` if at least one call started, and `"stop"` if none started. Usage is computed as before.

### Example 3

Suppose `tools` defines `get_weather` and `get_time`, and `tool_choice` is absent. The model's top choices spell 76 tokens followed by end-of-sequence:

```
Checking.<tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>
```

When streamed, the chunks between the role chunk and the finish chunk are:

| tokens | chunks |
|---|---|
| 1 to 9: `C` ... `.` | nine content chunks, one per token |
| 10 to 38: `<` ... `r` | none (the tag is held back and the name is incomplete) |
| 39: `"` | start of call 0: `id`, `"type": "function"`, `"name": "get_weather"`, `"arguments": ""` |
| 40 to 51 | none |
| 52 to 63: `{"`, `c`, `i`, `t`, `y`, `": `, `"`, `O`, `s`, `l`, `o`, `"}` | twelve argument fragments, one per token |
| 64 to 76: `}`, `<` ... `>` | none |

The finish chunk carries `"finish_reason": "tool_calls"`, and `completion_tokens` is 77. When not streamed, the message is:

```json
{"role": "assistant", "content": "Checking.",
 "tool_calls": [{"id": "call_6f1c...", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]}
```

With `"max_tokens": 22`, generation stops inside the tool name. In that case, `content` is `"Checking."`, no call starts, and `finish_reason` is `"length"`. With `"tool_choice": "none"` or without `tools`, the entire text is returned as `content` with `finish_reason` `"stop"`.

## Stage 4: cancellation and the compilation cache

**Client disconnect.** If the client closes the connection mid-stream, stop reading rows and close the model's stream. As in `sse-token-stream`, `send` does not raise an error on a closed connection under uvicorn. Instead, detect disconnects by awaiting `receive()` until it returns `{"type": "http.disconnect"}`. Non-streaming requests are not disconnected in the tests.

**Compilation cache.** Compiling a schema is the slowest step of a constrained request, and clients frequently reuse the same schemas. Each app returned by `make_app` must maintain its own cache of compiled schemas:

- Two schemas are identical if their parsed JSON representations produce the same compact JSON string with original key order preserved (`json.dumps(schema, separators=(",", ":"))`, or a hash of it). Request whitespace does not matter, but key order does because `properties` in a different order define a different JSON language.
- The first request needing a schema must call `CharDFA.from_regex` **exactly once** and build the **complete** token index for every state of the automaton before responding.
- Every subsequent request needing that same schema must make **zero** calls to `CharDFA.from_regex`, `dfa.step`, and `dfa.walk`, regardless of which tokens appear. `response_format` schemas and named-tool `parameters` schemas share the same cache, keyed solely by the schema itself (not by source field or `json_schema.name`).

## Constraints

- Request bodies are at most 64 KiB, generations have at most 1,000 rows, and compiled DFAs have at most a few hundred states. Compiling a test schema and indexing 282 tokens takes well under a second.
- **Guaranteed inputs.** Each row contains exactly `vocab.size` finite floats. The chosen token is never a special token other than end-of-sequence, and its bytes are always standalone valid UTF-8 (partial-character tokens are scored far below all others). Every reachable state has at least one candidate token.
- Responses are checked after JSON decoding (ignoring key order, spacing, and extra fields such as `"logprobs": null`). Tests verify content, tool calls, individual stream chunks, finish reason, usage, and the exact number of rows read. Request and call ids are checked for their required prefix and uniqueness.
- Completed constrained documents are also validated directly against the schema.
- Incremental streaming is checked by ordering rather than timing: the model is gated and yields row `i + 1` only after the test receives the chunk for token `i`.
- For disconnect tests, the model yields a row every 5 ms while the client reads a few chunks and disconnects. The test fails if the model is drained to completion or if its stream is not closed within 5 s.
- **Stage 4 counters.** The harness counts `CharDFA.from_regex` calls and `dfa.step` calls per request with zero tolerance: one compilation for a new schema, zero compilations and zero DFA steps for a cached schema. Each test uses fresh schemas not seen in other tests.
- Do not block the event loop with `time.sleep` or synchronous I/O.

## Why this matters

Structured output servers sit at the boundary between HTTP serving and token-level decoding. Agent frameworks parse every model reply as JSON or tool calls, so relying on prompts alone fails a few times in a thousand with stray prose or trailing commas and crashes downstream tools. Token masking guarantees conformance unconditionally.

Most real-world bugs appear at the seams between components. Applying a constraint on the non-streaming path while forgetting the streaming path passes basic tests and breaks production clients. Reporting `"stop"` on a response truncated by `max_tokens` tricks clients into parsing incomplete JSON. Repeating `id` or `name` in every streamed tool-call delta causes clients to open duplicate calls. Finally, recompiling schemas per request inflates time-to-first-token, keying the cache incorrectly serves the wrong automaton, and ignoring client disconnects wastes GPU slots on abandoned requests.
