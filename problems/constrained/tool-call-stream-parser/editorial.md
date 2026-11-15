## The idea

A streaming parser differs from a parser in one respect: it is interrupted after
every character and asked what it knows *for certain*. Two disciplines follow.

**All knowledge lives in the state.** If the meaning of the next character depends
on anything (are we in a tag, in a string, right after a backslash, how deep), that
thing is a field of the state, not a local variable and not something recomputed
from the current piece. Then processing `ab` in one step and `a`, `b` in two steps
pass through the same sequence of states, and chunking invariance is not something
to test for but a consequence of the structure. The simplest way to guarantee it is
to define the parser on single characters and let `parse_step` loop.

**Emit exactly what is certain.** Most characters are certain at once: a letter in
content mode is content, a character inside the arguments object belongs to the
arguments. The exception is a suffix of the content that is a proper prefix of
`<tool_call>`. Emitting it risks leaking a tag; holding more than that suffix delays
text for no reason. So after each character the parser keeps the longest suffix that
can still become the tag and emits everything before it. When a character breaks
the match, the held text is not all released blindly: the new, shorter suffix may
itself be a prefix of the tag (`<<tool_call>`, `<tool_<tool_call>`). This is the
failure function of Knuth-Morris-Pratt string matching, computed here by brute
force because the tag has eleven characters.

Inside a call the format does the work. The head is `{"name": NAME, "arguments":`,
so the second string literal is the name, complete at its closing quote, and the
first `{` outside a string after the third literal opens the arguments. From there
the parser is the string-and-depth core of a JSON recognizer: a quote toggles
"in string" unless the escape flag is set, a backslash in a string sets the flag for
exactly one character, and outside strings brackets move the depth. Depth zero is
the end of `ARGS`, however many braces, quotes, and tags its strings contain. Each
argument character is emitted the moment it is read, so the deltas concatenate to
the object as written and the client can parse it when the call ends.

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

`_emit` appends a delta, or extends the previous event when it is the same kind (and
the same call), which gives the merged form the statement asks for. The state is a
`NamedTuple`, so it cannot be modified by accident.

## Common mistakes

- **No holdback across pieces.** Searching each piece for `<tool_call>` works in
  every unit test that passes whole strings and fails in production, where the tag
  is several tokens. The partial tag is emitted as content and the call is never
  seen.
- **Flushing everything on a mismatch.** `<<tool_call>`: the buffer `<<` is not a
  prefix of the tag, so both characters are emitted and the tag that begins at the
  second `<` is missed. Keep the longest suffix that still matches.
- **Holding too much.** Buffering content until a tag or the end of the stream
  gives the right merged result and no streaming. The held text is at most ten
  characters.
- **Dropping the held text at the end.** A reply that ends with `<` (or with half a
  tag when the model was cut off by the token limit) loses its tail unless
  `parse_end` releases it.
- **Cumulative argument events.** Emitting "the arguments so far" at each step
  instead of the new characters. The OpenAI format is defined by concatenation; a
  client that appends cumulative fragments gets `{"city{"city": "Os...`.
- **Reusing index 0.** The index is how a client keeps parallel calls apart. With
  every call at index 0, the second call's name is ignored and its arguments are
  appended to the first's.
- **Counting braces without tracking strings**, or **searching for the end tag**.
  Argument values are arbitrary text: code with braces, a prompt that mentions
  `</tool_call>`. Only the string-aware depth count finds the real end.
- **An escape flag that does not survive the step.** A local `escaped = False` at
  the top of `parse_step` is right for every chunking except the ones that cut
  between `\` and `"`.
- **Mutating the state.** A state that is a mutable object updated in place breaks
  any caller that keeps one to retry or branch, and makes the function untestable
  for exactly the property this problem is about.
- **Reporting the name as written.** The name is a JSON string; `"get_x"`
  names the tool `get_x`.

## Complexity

Linear in the stream: each character is handled once, in constant time (the
content-mode suffix search is bounded by the tag length). The state is constant
size apart from the name being read and the few characters held or buffered in the
tail.

## In the wild

- **vLLM** implements this per model family; the parser for this format is its
  Hermes tool parser. Its streaming method is stateless per call in a different way:
  it is handed the full text so far and re-derives everything on each delta. Content
  is released up to the first `<tool_call>`, withholding a suffix that overlaps the
  beginning of the tag (a helper named `partial_tag_overlap`); the name is taken with
  a regular expression that requires its closing quote, so a partial name is never
  sent; and argument deltas are the difference between the arguments text seen now
  and the length already streamed. Earlier versions buffered and repaired partial
  JSON with a lenient parser instead. Re-parsing the whole text per token is
  quadratic in the reply length, which is the cost of not carrying a state machine.
- **llama.cpp**'s server parses the accumulated output with a parser that accepts
  partial input and sends the difference between successive parsed messages.
- On the wire, the first chunk of a call carries `index`, `id`, `type: "function"`,
  and `function.name` with empty `function.arguments`; later chunks carry `index`
  and a fragment of `function.arguments`. There is no explicit end event: the stream
  ends with `finish_reason: "tool_calls"`. `tool_call_end` exists here so that a
  server knows when it may start executing the call (see `speculative-tool-calls`).
- If the output is produced under a grammar (`structural-tag-dispatch`), the
  well-formedness this parser assumes is guaranteed by construction; without one, a
  production parser also needs a policy for malformed calls, which this problem
  leaves out.
