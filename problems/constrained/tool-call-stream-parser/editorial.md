## The idea

A streaming parser differs from an offline parser because it is interrupted after every piece and asked to emit everything it knows for certain. Two design rules make this clean and exact.

**Keep all parser knowledge in the state.** Whenever the meaning of the next character depends on prior context (whether we are inside a tag, inside a JSON string, right after a backslash, or at a particular brace depth), store that context as a field in the state rather than a local variable. Processing `"ab"` in one step then passes through the exact same sequence of states as processing `"a"` and `"b"` in two steps. Chunking invariance follows automatically from processing one character at a time.

**Emit everything that is certain and nothing more.** Most characters are certain immediately. A normal letter in content mode is content, and any character inside `ARGS` belongs to the arguments. The only ambiguity in content mode is a suffix that is a proper prefix of `<tool_call>`. Emitting it risks leaking a tag, while holding anything longer delays text unnecessarily. After each character, keep the longest suffix of the buffer that is a prefix of `<tool_call>` and emit everything before it. On a mismatch, do not blindly flush the whole buffer, because a shorter suffix may still start a tag (as in `<<tool_call>` or `<tool_<tool_call>`). This is the Knuth-Morris-Pratt failure function, computed here by brute force since the tag has only 11 characters.

Inside a tool call, the fixed structure handles the rest. The call head has the shape `{"name": NAME, "arguments":`, so the second completed JSON string literal is `NAME`, which finishes at its closing quote. The first `{` outside a string after the third literal (`"arguments"`) opens `ARGS` at depth 1. From there, track JSON strings with `in_string` and `escaped`, and track nesting with `depth`. A quote toggles `in_string` unless `escaped` is set, and a backslash inside a string sets `escaped` for the next character only. Outside strings, `{` and `[` increment `depth`, while `}` and `]` decrement it. When `depth` reaches 0, `ARGS` has ended regardless of any braces, quotes, or tags inside its strings. Emitting every character of `ARGS` as it arrives ensures the deltas concatenate to the exact object as written.

## Reference solution (Python)

```python
class State(NamedTuple):
    mode: str = "content"
    index: int = 0
    held: str = ""
    strings: int = 0
    depth: int = 0
    in_string: bool = False
    escaped: bool = False


def parse_step(state, piece):
    mode, index, held, strings, depth, in_string, escaped = state
    events = []
    for ch in piece:
        if mode == "content":
            held += ch
            if held == OPEN:
                mode, held, strings = "head", "", 0
                continue
            keep = len(held)
            while keep and not OPEN.startswith(held[len(held) - keep :]):
                keep -= 1
            if keep < len(held):
                _emit(events, "content", held[: len(held) - keep])
                held = held[len(held) - keep :]
        elif mode == "head":
            if in_string:
                held += ch
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                    strings += 1
                    if strings == 2:
                        events.append(("tool_call_start", index, json.loads(held)))
            elif ch == '"':
                in_string, held = True, ch
            elif ch == "{" and strings == 3:
                mode, depth, held = "arguments", 1, ""
                _emit(events, "arguments", index, ch)
        elif mode == "arguments":
            _emit(events, "arguments", index, ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    mode, held = "tail", ""
        else:
            held += ch
            if held.endswith(CLOSE):
                events.append(("tool_call_end", index))
                mode, index, held = "content", index + 1, ""
    return State(mode, index, held, strings, depth, in_string, escaped), events


def parse_end(state):
    if state.mode == "content" and state.held:
        return [("content", state.held)]
    return []
```

The helper `_emit` appends a new event, or merges into the previous event when it has the same kind (and same call index), producing the merged list required by the specification. Using a `NamedTuple` for `State` guarantees immutability.

## Common mistakes

- **No holdback across pieces.** Searching each piece independently for `<tool_call>` passes tests on whole strings and fails in production, where the tag spans multiple tokens. The partial tag leaks as content and the call is missed.
- **Flushing everything on a mismatch.** On `<<tool_call>`, the buffer `<<` is not a prefix of `<tool_call>`. Flushing both characters drops the second `<` that starts the real tag. Always keep the longest matching suffix.
- **Holding too much.** Buffering all content until a tag or end-of-stream produces the right merged text but breaks streaming. Held content must never exceed 10 characters.
- **Dropping held text at the end.** A stream ending with `<` or a truncated tag loses its tail unless `parse_end` flushes it.
- **Cumulative argument events.** Emitting all arguments seen so far instead of only the new characters breaks OpenAI-style concatenation, producing malformed text like `{"city{"city": "Os...`.
- **Reusing index 0.** Clients use `index` to separate multiple calls. Reusing `0` causes later calls' arguments to be appended to the first call.
- **Counting braces without tracking strings, or scanning for `</tool_call>`.** Argument values can hold arbitrary strings, including code with braces or literal `</tool_call>` tags. Only a string-aware depth counter finds the true end of `ARGS`.
- **Resetting `escaped` on every step.** A local `escaped = False` at the start of `parse_step` works for every split except one that falls between `\` and `"`.
- **Mutating the state.** Updating a mutable state object in place breaks callers that retain old states to branch or retry.
- **Reporting the raw name.** `NAME` is a JSON string literal, so `"get_x"` must be decoded to `get_x`.

## Complexity

Time is linear in the stream length. Each character is processed once in constant time, because the content-mode suffix check is bounded by the 11-character tag length. State memory is constant aside from the `NAME` string while reading the head and the short tail buffer.

## In the wild

- **vLLM** implements a separate tool parser per model family, using its Hermes tool parser for this format. Its streaming method receives the full accumulated text on each delta and re-derives the diff. It withholds tag prefixes via `partial_tag_overlap`, extracts `NAME` with a regex requiring the closing quote, and slices out new argument characters by length. Earlier versions instead repaired partial JSON on every step. Re-parsing the full text per token is quadratic in reply length, which is the cost of avoiding a state machine.
- **llama.cpp**'s server parses accumulated output with a partial-input parser and emits the diff between successive parsed messages.
- On the OpenAI wire format, the first chunk of a call carries `index`, `id`, `type: "function"`, `function.name`, and an empty `function.arguments`, while later chunks carry `index` and a fragment of `function.arguments`. There is no explicit end event on the wire (the stream finishes with `finish_reason: "tool_calls"`). `tool_call_end` is included here so a server knows when it can begin executing a completed call (see `speculative-tool-calls`).
- When generation is constrained by a grammar (`structural-tag-dispatch`), the well-formedness assumed here holds by construction. Without a grammar, production parsers also need fallback handling for malformed tool calls.
