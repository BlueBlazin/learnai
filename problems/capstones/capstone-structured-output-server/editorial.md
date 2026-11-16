## The idea

Four earlier problems, one request path. The design question is where the decision
"which token comes next" lives, because six combinations need it (text, a schema's
document, or a tool call; streamed or not), and a constraint that is implemented in
one of them and forgotten in another is the classic bug of this kind of server. The
answer that makes every stage short is three layers, each consuming the one before:

![One request through the server. The request is validated and its schema, if any, is looked up in the compilation cache or compiled once. The loop takes one row of logits per step from the model, masks it with the token index, and picks the best allowed token; the tool-call parser and the renderer turn the tokens into deltas, which are either streamed as chunks or folded into one message.](assets/architecture.svg)

1. **The generation loop** owns the model's stream and the automaton's state. It is
   the only code that looks at logits, and it yields token texts.
2. **The delta generator** turns token texts into the deltas of the response:
   content, the start of a tool call, an argument fragment.
3. **Two renderers** package the deltas: one chunk each for a stream, or folded into
   a single message.

**Stage 1** is `openai-chat-endpoint` with the model opened up. The scripted model
of that problem produced text and hid its stop token, so the finish reason had to be
inferred from a count. A real engine sees the stop token: `finish_reason` is `"stop"`
exactly when end-of-sequence was sampled, and `"length"` when the loop ran out of
budget first, which also settles the case that the count cannot (the limit reached
by the stop token itself). The loop must also *close* the model's stream when it
sees end-of-sequence. The model yields logits for as long as it is asked; "the
generation is over" is the server's knowledge.

**Stage 2** composes `json-schema-regex`, the kit's regex compiler, and
`dfa-token-index`. The loop changes in one line: the candidates are no longer all
tokens but `index[state]`, plus end-of-sequence when the state is accepting. Two
properties make this safe. The kit's automaton is *trimmed*, so "the walk did not
die" means "this is a prefix of some document", and a state with no way forward does
not exist. And end-of-sequence is offered only in accepting states, so the model
cannot stop inside a document; it can only be *cut off* there, and then the finish
reason says `"length"`. That is the whole guarantee of structured output: with
`"stop"`, the content is a document of the schema; with `"length"`, it is a prefix of
one and the client must not parse it.

Note what the constraint does to a model that disagrees with it. In the example the
model wants `": ` with a space; the mask removes it and the next best candidate,
`":`, wins. The model is not asked again and its later rows do not know. A real
model conditions on the token that was actually chosen, which is why constrained
decoding usually stays fluent, but the mechanism is the same: the server, not the
model, has the last word on every token.

**Stage 3** has two routes to a tool call, and they are different compositions.
With `"auto"` the model decides: the generation is unconstrained and the incremental
parser of `tool-call-stream-parser` reads the token texts; its events map one to one
onto deltas. With a named choice the server decides: the name is known before the
first token, so the start delta goes out at once, and the generation is constrained
by the tool's `parameters` exactly like a `response_format`, each token being an
argument fragment. Because the non-streaming message is defined as the fold of the
deltas, the two response modes cannot disagree about a tool call.

**Stage 4** adds the two things a server owes its neighbors. A stream is run as two
tasks, the sender and a watcher on `receive()`, because a send to a closed
connection does not fail under uvicorn; whichever finishes first ends the request,
and a `finally` closes the model's stream. And the compiled constraint (start
state, accepting states, the complete index) is cached in the closure of `make_app`
under the schema's JSON text. The key deserves the care: `id(schema)` never repeats,
since every request parses its own dict, and may repeat for a *different* schema
once an id is recycled; `sort_keys=True` merges schemas whose `properties` are
declared in a different order, which are different languages because members follow
declaration order. The index is built eagerly, for every state, so a cache hit
costs no automaton step at all.

## Reference solution (Python)

```python
class Generation:
    def __init__(self, rows, vocab, constraint=None):
        self.rows, self.vocab, self.constraint = rows, vocab, constraint
        self.count = 0  # rows consumed, end-of-sequence included
        self.ended = False  # True once end-of-sequence was chosen

    async def texts(self):
        eos, constraint = self.vocab.eos_id, self.constraint
        state = constraint.start if constraint else None
        async for row in self.rows:
            self.count += 1
            if constraint is None:
                candidates = range(len(row))
            else:
                candidates = list(constraint.moves[state])
                if state in constraint.accepting:
                    candidates.append(eos)
            token = max(candidates, key=lambda t: (row[t], -t))  # ties: the lowest id
            if token == eos:
                self.ended = True
                return
            if constraint is not None:
                state = constraint.moves[state][token]
            yield self.vocab.bytes(token).decode()


class Constraint:
    def __init__(self, schema, vocab):
        dfa = CharDFA.from_regex(schema_to_regex(schema))
        self.start, self.accepting = dfa.start, dfa.accepting
        self.moves = build_index(dfa, vocab)  # every step of the DFA is taken here


async def deltas(generation, request):
    if request.tool is not None:  # a named tool: the output is the arguments
        yield call_start(0, request.tool["name"])
        async for text in generation.texts():
            yield call_arguments(0, text)
    elif request.parse_tools:  # "auto": the model writes <tool_call> markup
        state = ParserState()
        async for text in generation.texts():
            state, events = parse_step(state, text)
            for event in events:
                yield event_delta(event)
        for event in parse_end(state):
            yield event_delta(event)
    else:
        async for text in generation.texts():
            yield {"content": text}


def finish_reason(generation, called):
    if not generation.ended:
        return "length"
    return "tool_calls" if called else "stop"


def make_app(lm, vocab):
    compiled = {}

    def constraint_for(schema):
        key = json.dumps(schema, separators=(",", ":"))  # key order kept
        if key not in compiled:
            compiled[key] = Constraint(schema, vocab)
        return compiled[key]

    async def app(scope, receive, send):
        ...  # route; parse_request; constraint_for(...) with ValueError -> 400
        rows = lm.stream(prompt, request.max_tokens)
        generation = Generation(rows, vocab, constraint)
        try:
            if request.stream:
                await stream_completion(generation, request, base, prompt_tokens, receive, send)
            else:
                await complete(generation, request, base, prompt_tokens, send)
        finally:
            await rows.aclose()

    return app
```

`schema_to_regex`, `build_index`, and the parser are those of the three
prerequisites, unchanged; `stream_completion` is the pump and watcher of
`openai-chat-endpoint` with `async for delta in deltas(...)` in the middle, and
`complete` applies the same deltas to a message dict.

## Common mistakes

- **The constraint on one path only** (`stream_unconstrained`). Two loops, one per
  response mode, and only one got the mask. One loop under both renderers makes the
  bug impossible.
- **A truncated document reported as finished** (`constrained_finish_stop`).
  "Constrained output is always valid" is true only when the model was allowed to
  finish. A prefix is not a document.
- **End-of-sequence offered everywhere** (`eos_in_any_state`). Then a model that
  wants to stop ends the document in the middle, with `finish_reason: "stop"`.
- **Inferring the finish reason from the count** (`finish_from_count`), and
  **not counting the stop token** (`eos_not_counted`): both are habits from a model
  that hides its stop token.
- **Leaving the model running after end-of-sequence** (`stream_left_open`), or
  after the client has gone (`ignores_disconnect`). The model has no idea that
  nobody is listening.
- **Ties broken the other way** (`ties_highest_id`), and **buffering the stream**
  (`buffers_stream`).
- **A 500 for an unsupported schema** (`unsupported_schema_500`): compile during
  validation, where a `ValueError` can still become a `400`.
- **Tool calls with `finish_reason: "stop"`** (`tool_finish_stop`): an agent loop
  decides whether to run tools by this field.
- **Repeating id and name in every fragment** (`id_in_every_delta`): clients open a
  call when they see an id.
- **A named tool without the constraint** (`named_choice_unconstrained`), and
  **parsing markup when tools are off** (`parses_without_tools`).
- **A cache keyed by object identity** (`cache_by_identity`) or **by sorted keys**
  (`cache_sorted_keys`); **caching the automaton but not the index**
  (`index_per_request`); **filling the index lazily** (`lazy_index`), which a real
  engine may choose on purpose, but which moves compile time into later requests,
  where this problem's counter sees it.
- **Borrowing the kit's `DFAConstraint`** (`uses_kit_constraint`): the index is the
  part of the pipeline you are asked to own.

## Complexity

Compiling a schema costs the regex compilation plus one automaton step per state
and distinct vocabulary prefix that is still alive (the trie walk of
`dfa-token-index`): tens of thousands of steps here, milliseconds, and on a real
vocabulary of 100,000 tokens and an automaton of thousands of states, seconds. That
is why it is cached. A generation step is then a dictionary lookup and a maximum
over the allowed tokens, $O(V)$ at worst and usually a handful. The cache holds one
entry per distinct schema; a production server bounds it (LRU) and compiles off the
event loop.

## In the wild

The request and response shapes are those of the OpenAI Chat Completions API
(`response_format` of type `json_schema` with a required `name`; `delta.tool_calls`
entries with `index`, and `id`, `type`, and `function.name` on the first fragment
only; the finish reasons `stop`, `length`, `tool_calls`), checked against the type
definitions of the `openai-python` SDK. The regex-to-index pipeline is the method
of "Efficient Guided Generation for Large Language Models" (arXiv 2307.09702), as
implemented in Outlines; vLLM exposes it, and grammar-based alternatives such as
XGrammar and llguidance, behind the same request fields
(https://docs.vllm.ai/en/latest/features/structured_outputs/). Two simplifications to know about:
real servers render `tools` into the prompt through the chat template, and the
markup a model uses for a call (`<tool_call>` here) differs per model family, which
is why engines ship one parser per family.
