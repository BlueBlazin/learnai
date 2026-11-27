## The idea

All six combinations of output mode (plain text, schema-constrained JSON, or tool calls) and delivery mode (streaming or non-streaming) need the same core token selection. Implementing them in separate loops makes it easy to forget a constraint or finish-reason rule on one path. A clean way to avoid that is to stack three layers, each consuming the one below it:

![The path of one request. The token index masks each row of logits inside the loop, and the parser turns the chosen tokens into deltas.](assets/architecture.svg)

1. **Generation loop.** Owns the model stream and the DFA state. It is the only layer that inspects logits, and it yields decoded token strings.
2. **Delta generator.** Converts token strings into response deltas: content pieces, tool-call starts, and argument fragments.
3. **Renderers.** Either stream each delta as an SSE chunk or fold all deltas into a single non-streaming message.

**Stage 1 (chat completions).** Unlike `openai-chat-endpoint`, where the scripted model hid its stop token and forced you to infer `finish_reason` from the token count, here end-of-sequence is an explicit token in the logits. `finish_reason` is `"stop"` if and only if end-of-sequence was chosen, and `"length"` if the iterator ran out of rows first. That also handles the boundary case where the final allowed row selects end-of-sequence. Because the model never knows which token the server picked, the server must explicitly close the stream once end-of-sequence is chosen.

**Stage 2 (schema-constrained output).** Composing `json-schema-regex`, `CharDFA.from_regex`, and `dfa-token-index` changes only one line of the generation loop. Instead of all tokens, the candidates are `index[state]`, plus `vocab.eos_id` when `state` is accepting. Two properties guarantee correctness. First, the kit's DFA is trimmed, so every live transition can still reach an accepting state and dead ends cannot occur. Second, end-of-sequence is allowed only in accepting states, so the model cannot stop mid-document. It can only be cut off by `max_tokens`, in which case `finish_reason` is `"length"` and signals to the client that the output is only a prefix.

In Example 2, the model ranks `": ` highest, the mask excludes it, and the next best allowed token `":` wins. A real LLM conditions future steps on the chosen token to stay fluent, while this scripted model ignores previous choices, but the masking mechanism is identical.

**Stage 3 (tool calls).** `"auto"` and named tool choices compose the layers differently. Under `"auto"`, generation is unconstrained and `tool-call-stream-parser` translates token strings into events that map directly to deltas. Under a named choice, the server already knows the tool name, so it emits the start delta immediately, constrains generation with the tool's `parameters` schema, and emits every chosen token as an argument fragment. Because the non-streaming renderer folds the exact same delta stream, streaming and non-streaming responses cannot disagree.

**Stage 4 (cancellation and caching).** Under uvicorn, sending to a closed socket does not raise an error. Run the stream sender and a `receive()` disconnect watcher as two concurrent tasks, cancel both as soon as either finishes, and close the model stream inside a `finally` block. Store compiled constraints (start state, accepting states, and the full token index) in a dict inside `make_app`, keyed by `json.dumps(schema, separators=(",", ":"))`. Avoid keying by `id(schema)` (every parsed request creates a new dict, and Python recycles ids) or using `sort_keys=True` (reordering `properties` changes the allowed JSON key order). Building the index eagerly for every DFA state ensures cache hits perform zero DFA steps.

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

`schema_to_regex`, `build_index`, and the tool-call parser are reused unchanged from their prerequisite problems. `stream_completion` runs the sender and disconnect watcher from `openai-chat-endpoint` around `async for delta in deltas(...)`, and `complete` folds those same deltas into a single message dict.

## Common mistakes

- **Applying the constraint on one path only (`stream_unconstrained`).** Duplicating the decoding loop for streaming and non-streaming makes it easy to mask logits in only one place. Sharing one `Generation` loop eliminates this bug.
- **Reporting `"stop"` on a truncated document (`constrained_finish_stop`).** A constrained response is a complete document only when end-of-sequence was chosen. A budget-truncated prefix must report `"length"`.
- **Allowing end-of-sequence in non-accepting states (`eos_in_any_state`).** Offering EOS everywhere lets the model stop mid-document while still reporting `"stop"`.
- **Inferring finish reason from count (`finish_from_count`) or omitting EOS from usage (`eos_not_counted`).** Because EOS is explicitly sampled here, track whether EOS occurred and count its row in `completion_tokens`.
- **Leaving the model stream open (`stream_left_open`, `ignores_disconnect`).** Always close `lm.stream` in a `finally` block both after EOS and when `receive()` signals `http.disconnect`.
- **Wrong tie-breaking (`ties_highest_id`) or buffering chunks (`buffers_stream`).** Break logit ties toward the lowest token id and send each SSE chunk before requesting the next row.
- **Returning 500 on an unsupported schema (`unsupported_schema_500`).** Compile and validate the schema during request validation so a `ValueError` becomes a `400`.
- **Reporting `"stop"` when tool calls occurred (`tool_finish_stop`).** Agent loops inspect `finish_reason == "tool_calls"` to decide whether to invoke tools.
- **Repeating `id` and `name` in every fragment (`id_in_every_delta`).** Clients treat a delta with an `id` as a new tool call. Emit `id`, `type`, and `name` only in the start delta.
- **Unconstrained named choice (`named_choice_unconstrained`) or parsing markup when tools are off (`parses_without_tools`).** Enforce the named tool's `parameters` schema, and disable `<tool_call>` parsing when `tool_choice` is `"none"`.
- **Broken schema caching (`cache_by_identity`, `cache_sorted_keys`, `index_per_request`, `lazy_index`).** Key the cache by key-order-preserving compact JSON, cache the full token index alongside the DFA, and build all states eagerly so cache hits take zero DFA steps.
- **Using the kit's `DFAConstraint` (`uses_kit_constraint`).** The harness forbids `DFAConstraint` and DFA oracle methods so that your own token index does the work.

## Complexity

Compiling a schema requires regex-to-DFA compilation plus one DFA step per state and live vocabulary prefix (via the trie walk of `dfa-token-index`). That takes milliseconds on a 282-token vocabulary, but seconds on a 100,000-token production vocabulary with thousands of DFA states, which is why caching is essential. Once indexed, each generation step is a dictionary lookup and a maximum over allowed tokens, taking $O(V)$ time for a vocabulary of size $V$ in the worst case and typically far less. The cache stores one entry per distinct schema. Production servers bound cache memory with an LRU policy and run compilation off the event loop.

## In the wild

The request and response schemas follow the OpenAI Chat Completions API (`response_format` with `json_schema` and required `name`, `delta.tool_calls` with `index` plus `id`/`type`/`function.name` on the start delta only, and finish reasons `stop`, `length`, and `tool_calls`), matching the `openai-python` SDK types. The regex-to-DFA-to-token-index pipeline comes from "Efficient Guided Generation for Large Language Models" (arXiv:2307.09702), pioneered by Outlines and exposed alongside grammar-based engines like XGrammar and llguidance in vLLM (https://docs.vllm.ai/en/latest/features/structured_outputs/). Real servers add two pieces omitted here: they render `tools` into the prompt via the model's chat template, and they ship model-specific parsers because tool-call markup varies across model families.
