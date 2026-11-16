Both response modes, all three kinds of output (text, a schema's document, a tool
call), and the disconnect need the same thing from the model: the next chosen token.
Where does the decision "which token" live so that none of the six combinations can
forget the constraint? And what is a non-streamed message, if a chunk is a delta to
it?

---

Three layers, each a consumer of the one before. (1) A generation loop that reads a
row, builds the candidates (`range(vocab.size)`, or `index[state]` plus
`vocab.eos_id` when `state in dfa.accepting`), takes `max(candidates, key=lambda t:
(row[t], -t))`, counts the row, and stops at end-of-sequence; it yields token texts
and remembers `count` and whether it ended by EOS. (2) A delta generator on top of
it: `{"content": text}`; or, for `"auto"`, the events of `parse_step` mapped to
deltas (and `parse_end` after the loop); or, for a named tool, one start delta and
then an arguments delta per token. (3) Two renderers: the stream sends one chunk per
delta, the other applies the same deltas to `{"role": "assistant", "content": ""}`.
The finish reason is `"length"` if the loop did not see EOS, else `"tool_calls"` if
any delta had `tool_calls`, else `"stop"`. Close the model's stream in a `finally`
around both renderers.

---

Compile inside `make_app`, where the cache lives:

```python
compiled = {}


def constraint_for(schema):
    key = json.dumps(schema, separators=(",", ":"))  # key order kept: do not sort
    if key not in compiled:
        dfa = CharDFA.from_regex(schema_to_regex(schema))  # ValueError -> 400
        compiled[key] = (dfa.start, dfa.accepting, build_index(dfa, vocab))
    return compiled[key]
```

`build_index` is the one of `dfa-token-index`, for every state in `dfa.states`, so
that no later request steps the automaton. Resolve the constraint during validation,
before `lm.stream` is called: the schema of `response_format`, or the `parameters`
of the named tool (`{"type": "object", "properties": {}}` if it has none). For the
disconnect, run the sending coroutine and a `receive()` watcher as two tasks, wait
for the first to finish, cancel both, and let the `finally` close the model's stream.
