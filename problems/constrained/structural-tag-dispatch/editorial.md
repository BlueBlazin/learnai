## The idea

The language "free text, then a tag, then free text" is regular whenever the tag
contents are, so a dispatcher is a finite automaton over bytes, and a token is
allowed when the automaton can read all of its bytes. What makes the problem worth
doing is that the automaton is assembled from parts with different owners (the
triggers, the begin strings, the content automata, the end strings), and each
junction has a classic way to go wrong.

**Free text is a string search that never ends.** The reader has to notice the first
moment the text ends with a trigger, reading one byte at a time. The only part of the
past that can matter is the longest suffix that is still a proper prefix of some
trigger: anything before it can no longer be part of a match. That suffix is the
whole state. On a byte, extend it; if the result is not a prefix of any trigger,
drop bytes from the front until it is (possibly until it is empty). This is the
Aho-Corasick automaton of the trigger set, with the failure links computed on the
fly. Clearing the suffix on a mismatch is the tempting shortcut, and it is wrong
exactly when a trigger overlaps its own beginning.

Requiring that no trigger contains another makes "the first position where the text
ends with a trigger" unambiguous: if a trigger ended inside the held suffix earlier
than at its end, it would be a substring of the trigger the suffix belongs to.

**Dispatch is a narrowing set.** After the trigger, the tags that begin with it are
all candidates, and each byte keeps those that agree. Because no `begin` is a prefix
of another, the moment one is complete it is the only candidate, and the dispatch is
decided. A trigger without tags gives an empty candidate set at once, which makes it
a forbidden string for free: the engine's way to say "the model may write anything,
except this".

**Content hands over to the end string.** The content automaton reads for as long as
it can. When it cannot, the byte must begin the end string, and only if the content
is complete. The guarantee in the constraints (no accepting state can read the end
string's first byte) is what keeps this deterministic; without it the reader would
have to follow both possibilities at once, as `grammar-token-mask` does.

**Tokens see none of this.** A token is a few bytes, and those bytes may end a
trigger, pass through a begin string, and enter the content, or close the content
and the tag in one piece. So nothing is ever decided per token: the token mask is
"walk the bytes, see whether the walk survives", and the state after a token is
wherever the walk stopped. A special token has no bytes, survives every walk
vacuously, and has to be excluded by hand. End of sequence is the loop's business:
it is permitted exactly when `is_accepting` holds, which here means the reader is in
free text.

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

`_walk` applies `_step` to each byte and stops at the first `None`; `advance` is
`_walk` with a `ValueError`; `is_accepting` is `state[0] == "free"`. The constructor
encodes the strings to bytes and checks the three ambiguity rules. States are tuples
of ints, bytes, and tuples, so they are hashable, and nothing about a generation
lives in the object: two generations can share one dispatcher.

## Common mistakes

- **Looking for the trigger inside one token.** `<tool_call>` is a single token only
  in tokenizers that were trained with it as a special token. Everywhere else it is
  five to eight tokens, and a dispatcher that tests `trigger in piece` never fires.
  The partial match has to live in the state.
- **Clearing the partial match on a mismatch.** Correct for triggers that do not
  overlap themselves and wrong for `<<fn=` after `<<<`. Restart from the longest
  suffix that still matches.
- **Carrying the scan across a tag.** After the end string the scan starts from
  nothing. If the end string's last bytes are kept (a fence closed by three
  backticks, a trigger that begins with three backticks), ordinary text after the
  tag fires a trigger that is not there.
- **Taking the first tag of a trigger**, or **not narrowing the candidates.** The
  first makes all but one function of `<function=` uncallable; the second accepts
  the beginning of one tag followed by the rest of another.
- **Keeping content state in the object.** The content automaton must start from its
  start state for every call. State kept on `self` and never reset makes the second
  call continue the first one's content, and makes the dispatcher unusable for two
  requests at once.
- **Skipping the end string.** Returning to free text as soon as the content is
  complete lets the model leave the tag open; the client's parser, which waits for
  `</tool_call>`, then swallows the rest of the reply.
- **Accepting inside a tag.** `is_accepting` must be false from the byte that fires
  the trigger to the last byte of the end string, including the begin and end
  strings, not only the content.
- **Special tokens.** A walk over no bytes always survives. Without `if piece`,
  end-of-sequence is "allowed" in the middle of the arguments.
- **Ignoring a trigger without tags.** It is not a no-op: it is the one string the
  free text may not contain.
- **No validation.** With one trigger inside another, which one fires depends on the
  implementation; rejecting the configuration is the only honest answer.

## Complexity

One `allowed` call walks every token once: $O(\sum_t |t|)$ steps, each constant
apart from the trigger-prefix test, which is bounded by the number and length of the
triggers. An engine would precompute the failure links and, since the whole thing is
a finite automaton, index the allowed tokens per state as in `dfa-token-index`. In
free text almost every token is allowed, so real engines special-case it: only the
tokens that could touch a trigger need checking.

## In the wild

- **XGrammar** calls the format a structural tag. Its documentation describes
  `triggered_tags` as allowing "arbitrary text until a trigger is encountered", then
  dispatching "to one of several tags", with arbitrary text allowed again after the
  tag ends; each tag has a `begin`, a content specification, and an `end`, each tag
  should match exactly one trigger, and each trigger should be an unambiguous prefix
  of the tags it dispatches to. The validation rules of this problem are that
  requirement made checkable. Options there that this problem leaves out:
  `at_least_one` (the output must begin with a tag), `stop_after_first`, and
  `excludes`, strings the free text may not contain, which is what a trigger without
  tags amounts to here.
- The technical report "XGrammar-2: Dynamic and Efficient Structured Generation
  Engine for Agentic LLMs" (arXiv 2601.04426) names the dispatching construct
  TagDispatch.
- **vLLM** and other OpenAI-compatible servers accept the format as
  `response_format` of type `structural_tag` and pass it to the grammar backend, with
  JSON schemas for the tag contents instead of the regular expressions used here.
- The stream parser of `tool-call-stream-parser` is the other half of the same
  design: this constraint guarantees that what is generated after a trigger is a
  well-formed call, and the parser turns it into API events. Its "hold back a
  possible tag start" is the same suffix that the free state carries here.
