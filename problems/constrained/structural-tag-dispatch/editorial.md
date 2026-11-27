## The idea

Because each tag's content language is regular, the combined language of free text interspersed with tags is also regular. A dispatcher is therefore a finite automaton over bytes, and a token is allowed whenever the automaton can consume all of its bytes without failing. The main challenge is handling the transitions between the four components (triggers, begin strings, content automata, and end strings) cleanly.

**Free text is a continuous string search.** Reading one byte at a time, the reader must detect the first position where the current free-text buffer ends with a trigger. Only the longest suffix that is still a proper prefix of some trigger can participate in a future match, so that suffix is the entire free-text state. On each new byte, append it to the suffix. If the result is not a prefix of any trigger, drop bytes from the front until it is (or becomes empty). This mirrors an Aho-Corasick automaton with failure links computed on the fly. Clearing the suffix on a mismatch is a common shortcut, but it fails whenever a trigger overlaps its own beginning.

Forbidding one trigger from containing another makes the first match unambiguous. If another trigger ended earlier inside the held suffix, it would be a substring of the trigger that the suffix belongs to.

**Dispatch narrows a candidate set.** Once a trigger fires, every tag whose `begin` starts with that trigger becomes a candidate. Each subsequent byte keeps only the candidates that expect that byte next. Because no `begin` is a prefix of another, the first time a candidate's `begin` completes, it is guaranteed to be the only remaining candidate. A trigger with no belonging tags immediately produces an empty candidate set, which makes it a forbidden string in free text.

**Content hands off to the end string.** The `inner` automaton consumes bytes as long as it has a valid transition. When `inner.step` returns `DEAD`, the byte must match `end[0]` and the automaton must currently be in an accepting state. Because the constraints guarantee that no accepting state of `inner` transitions on `end[0]`, this handoff is deterministic. Without that guarantee, the reader would have to track both branches in parallel, as in `grammar-token-mask`.

**Tokens are transparent to the automaton.** A single token can end a trigger, traverse a `begin` string, and enter content, or close both the content and the `end` string at once. Nothing should be decided at the token level. To test whether a token is allowed, walk its bytes through the byte-level transition function and check that the walk succeeds. The state after `advance` is simply the state reached at the end of the walk. Special tokens have empty byte strings, so a byte walk would succeed vacuously and they must be filtered out explicitly. End of sequence is handled by the decoding loop, which allows stopping only when `is_accepting` is true (that is, in free text).

## Reference solution (Python)

```python
def _step(self, state, byte):
    kind = state[0]
    if kind == "free":
        held = state[1] + bytes([byte])
        while held and not any(trigger.startswith(held) for trigger in self.triggers):
            held = held[1:]
        if held not in self.triggers:
            return ("free", held)
        candidates = tuple(i for i, (begin, _, _) in enumerate(self.tags) if begin.startswith(held))
        return self._begun(candidates, len(held))
    if kind == "begin":
        _, candidates, matched = state
        candidates = tuple(i for i in candidates if self.tags[i][0][matched] == byte)
        return self._begun(candidates, matched + 1)
    if kind == "inner":
        _, tag, q = state
        _, inner, end = self.tags[tag]
        after = inner.step(q, byte)
        if after != DEAD:
            return ("inner", tag, after)
        if inner.is_accepting(q) and byte == end[0]:
            return self._ended(tag, 1)
        return None
    _, tag, matched = state
    if byte == self.tags[tag][2][matched]:
        return self._ended(tag, matched + 1)
    return None


def _begun(self, candidates, matched):
    if not candidates:
        return None
    first = candidates[0]
    if len(self.tags[first][0]) == matched:
        return ("inner", first, self.tags[first][1].start)
    return ("begin", candidates, matched)


def _ended(self, tag, matched):
    if matched == len(self.tags[tag][2]):
        return ("free", b"")
    return ("end", tag, matched)


def allowed(self, state):
    return [
        t for t, piece in enumerate(self.pieces) if piece and self._walk(state, piece) is not None
    ]
```

`_walk` applies `_step` to each byte of a token and returns `None` on the first invalid byte. `advance` calls `_walk` and raises `ValueError` if it returns `None`. `is_accepting` checks `state[0] == "free"`. The constructor encodes all strings to UTF-8 bytes and checks the three ambiguity rules. Because states are tuples of strings, integers, bytes, and tuples, they are hashable and immutable, and multiple generations can safely share one dispatcher instance.

## Common mistakes

- **Looking for the trigger inside a single token.** A trigger like `<tool_call>` is one token only if the tokenizer added it as a special token. Otherwise it spans five to eight tokens, so testing `trigger in piece` never fires. The partial trigger match must live in the state.
- **Clearing the partial match on a mismatch.** Resetting the suffix to empty works only when triggers do not overlap themselves. For a trigger like `<<fn=`, reading `<<<` must retain `<<` rather than clearing the buffer.
- **Carrying the trigger scan across a tag.** After a tag's `end` string finishes, free text starts fresh with an empty suffix. Keeping trailing bytes from `end` (such as a closing triple backtick when a trigger starts with three backticks) causes ordinary text after the tag to fire a false trigger.
- **Picking the first matching tag or failing to narrow candidates.** Picking the first tag of a trigger makes every other tag sharing `<function=` unreachable. Failing to narrow the candidate set accepts the prefix of one tag followed by the suffix of another.
- **Storing content state on `self`.** Each tag invocation must start its `inner` automaton at `inner.start`. Storing mutable state on `self` corrupts subsequent calls and breaks concurrent generations sharing the dispatcher.
- **Skipping the end string.** Returning to free text as soon as `inner` accepts lets the model leave tags unclosed. A downstream parser waiting for `</tool_call>` then swallows the rest of the reply.
- **Accepting inside a tag.** `is_accepting` must be `False` from the byte that fires the trigger through the last byte of `end`, including the `begin` and `end` phases, not only during `inner`.
- **Allowing special tokens.** Walking zero bytes always succeeds. Without checking `if piece`, end-of-sequence tokens appear in `allowed` even inside a tag.
- **Ignoring a trigger with no tags.** Such a trigger is not a no-op. It represents a string that free text is forbidden from containing.
- **Skipping constructor validation.** If one trigger contains another, which trigger fires becomes implementation-dependent. Rejecting ambiguous configurations upfront is the only consistent behavior.

## Complexity

Each call to `allowed` walks every token in the vocabulary once, taking $O(\sum_t |t|)$ byte steps (where $|t|$ is the byte length of token $t$). Each step is constant time except for the trigger-prefix check in free text, which is bounded by the number and length of the triggers. A production engine precomputes Aho-Corasick failure links and indexes allowed tokens per automaton state (as in `dfa-token-index`). Because almost every token is allowed in free text, production engines also special-case free text and inspect only tokens that could touch or advance a trigger.

## In the wild

- **XGrammar** calls this construct a structural tag. Its `triggered_tags` format allows arbitrary text until a trigger appears, dispatches to one of several tags (each with `begin`, content schema, and `end`), and returns to arbitrary text after `end`. XGrammar requires each tag to match one trigger and each trigger to be an unambiguous prefix of its tags, which matches the validation rules here. XGrammar also supports options omitted in this problem, including `at_least_one` (requiring the output to start with a tag), `stop_after_first`, and `excludes` (forbidden strings in free text, equivalent to a trigger with no tags here).
- The technical report "XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs" (arXiv 2601.04426) refers to this dispatching construct as `TagDispatch`.
- **vLLM** and other OpenAI-compatible serving engines expose this feature via `response_format` with type `structural_tag`, passing JSON schemas for the tag contents to the grammar backend instead of the regular expressions used here.
- The stream parser in `tool-call-stream-parser` is the decoding-side counterpart to this problem. This constraint guarantees that text after a trigger forms a valid tool call, while the stream parser converts it into structured API events. Its buffered prefix for a potential tag start is the exact same suffix that the free-text state tracks here.
