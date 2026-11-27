All six combinations of response mode (streaming or non-streaming) and output type (plain text, schema-constrained JSON, or tool calls) share the same core step: picking the next token. Where should token selection live so none of the six paths can accidentally skip the constraint? And how does a non-streaming message relate to a stream of deltas?

---

Stack three layers, each consuming the one below it:

1. **Generation loop:** reads each row of logits, builds the candidate set (`range(vocab.size)`, or `index[state]` plus `vocab.eos_id` when `state in dfa.accepting`), picks `max(candidates, key=lambda t: (row[t], -t))`, increments `count`, and stops on `vocab.eos_id`. It yields decoded token texts and records `count` and whether EOS was seen.
2. **Delta generator:** wraps the generation loop. It yields `{"content": text}` for plain text or JSON schema output, maps `parse_step` and `parse_end` events to deltas for `"auto"`, or emits one start delta followed by one argument delta per token for a named tool.
3. **Two renderers:** the streaming renderer sends one SSE chunk per delta, while the non-streaming renderer folds those same deltas into `{"role": "assistant", "content": ""}`.

Set `finish_reason` to `"length"` if the loop did not see EOS, `"tool_calls"` if any delta included `tool_calls`, and `"stop"` otherwise. Close `lm.stream` in a `finally` block wrapping both renderers.

---

Keep the schema cache inside `make_app`:

```python
compiled = {}


def constraint_for(schema):
    key = json.dumps(schema, separators=(",", ":"))  # key order kept: do not sort
    if key not in compiled:
        dfa = CharDFA.from_regex(schema_to_regex(schema))  # ValueError -> 400
        compiled[key] = (dfa.start, dfa.accepting, build_index(dfa, vocab))
    return compiled[key]
```

Use `build_index` from `dfa-token-index` across all `dfa.states` eagerly so later requests never step the DFA. Resolve the constraint during request validation (before calling `lm.stream`) using either `response_format`'s schema or the named tool's `parameters` (`{"type": "object", "properties": {}}` if omitted). To handle disconnects, run the SSE sender and a `receive()` watcher as two asyncio tasks, wait for the first to finish, cancel both, and let the `finally` block close the model stream.
