An assistant that can call tools writes ordinary prose most of the time and a
rigidly structured call now and then. Constraining the whole output to a grammar is
wrong for the prose, and leaving it unconstrained is wrong for the call. A structural
tag does both: the output is free until the model writes a *trigger* string; from
that moment it must complete one of the tags that begin with the trigger, with
content in the tag's own language, up to the tag's end string; then it is free
again. Implement this dispatcher as a token constraint. The triggers, the tags, and
the tokens all have their own boundaries, and none of them line up.

## Specification

```python
class TagDispatch:
    def __init__(self, triggers, tags, vocab): ...
    def start(self): ...  # the state before any token
    def allowed(self, state) -> list[int]: ...
    def advance(self, state, token): ...  # the state after `token`
    def is_accepting(self, state) -> bool: ...
```

This is the `TokenConstraint` protocol of `constrained-greedy-decode`.

- `triggers`: a list of non-empty strings.
- `tags`: a list of tuples `(begin, inner, end)`. `begin` and `end` are strings and
  `inner` is a [`CharDFA`](../../../python/src/whetstone/kit/dfa.py), built with
  `CharDFA.from_regex`, for the language of the tag's content. A tag *belongs to* the
  trigger that its `begin` starts with.
- `vocab`: a [`ToyVocab`](../../../python/src/whetstone/kit/vocab.py);
  `vocab.bytes(t)` is the byte string of token `t`, empty for special tokens.

Everything below is about **bytes**: strings mean their UTF-8 encoding, and the
output is the concatenated bytes of the generated tokens.

### The language

Read the output from left to right. At any moment the reader is outside tags, in
*free text*, or inside one.

1. **Free text.** Any bytes are valid. The reader watches for the first position at
   which the free text read since it began (at the start of the output, or right
   after the last tag ended) ends with a trigger. Bytes before the current stretch of
   free text never count toward a trigger. At that position the trigger *fires*.
2. **Begin.** After a trigger has fired, the output must continue so that the
   trigger and the bytes after it spell the `begin` of one of the trigger's tags.
   While several tags are still consistent with what has been read, all of them
   remain candidates; a byte that no candidate has next is invalid. A trigger to
   which no tag belongs can therefore never be completed: it is a string the free
   text may not contain.
3. **Content.** When a `begin` is complete, the following bytes are read by that
   tag's `inner` automaton, from its start state: `inner.step(state, byte)` returns
   the next state, or `DEAD` (`-1`). The content goes on for as long as the
   automaton can read the next byte.
4. **End.** If the automaton cannot read the next byte, that byte must be the first
   byte of the tag's `end`, and the automaton must be in an accepting state
   (`inner.is_accepting(state)`); otherwise the byte is invalid. The rest of `end`
   must follow exactly. After its last byte the reader is in free text again,
   starting afresh.

An output is **valid so far** if no byte was invalid. It is **complete** if it is
valid so far and the reader is in free text: outside every tag. Free text may stop
anywhere, also where a trigger has only partly appeared, since that is not yet a
trigger.

### The methods

Let `text(state)` be the bytes of the tokens advanced through since `start()`.

- **`allowed(state)`** returns, in increasing order, the ids of all tokens `t` such
  that `vocab.bytes(t)` is not empty and `text(state) + vocab.bytes(t)` is valid so
  far. All of the token's bytes are read, whatever they cross: one token may contain
  the end of a trigger and the beginning of a `begin`, or the last byte of the
  content and the whole `end`. Special tokens are never in the list.
- **`advance(state, token)`** returns the state for `text(state) + vocab.bytes(token)`
  and raises `ValueError` if `token` is not allowed in `state`.
- **`is_accepting(state)`** returns whether `text(state)` is complete.

States are opaque to the caller, hashable, and never modified.

**The constructor** raises `ValueError` if the configuration is ambiguous:

- a trigger is contained in another trigger as a substring (this includes a trigger
  that is a prefix of another, and two equal triggers): the contained one would
  always fire first, or both at once;
- a tag's `begin` starts with no trigger;
- a tag's `begin` is a prefix of another tag's `begin` (or equal to it): after the
  shorter one it would be undecidable which tag has begun.

Otherwise each `begin` starts with exactly one trigger, and several tags may belong
to the same trigger.

Learn the automata only through `step` and `is_accepting`. Their test oracles
`accepts`, `accepts_prefix`, and `is_live` are not allowed (the harness rejects
them).

## Example

```python
vocab = ToyVocab(size=132, seed=0)  # every printable ASCII character has a token
dispatch = TagDispatch(
    triggers=['{"fn"'],
    tags=[
        ('{"fn": "neg", "x": ', CharDFA.from_regex(r"-?\d+"), "}"),
        ('{"fn": "say", "text": ', CharDFA.from_regex(r'"[a-z ]*"'), "}"),
    ],
    vocab=vocab,
)
```

The vocabulary has 127 tokens with bytes: a token for each printable character, and
longer pieces such as `{"`, `":`, `": `, `"}`, `, `, `true`, `nul`.

| text so far | where the reader is | allowed | accepting |
|---|---|---|---|
| (empty) | free text | all 127 | yes |
| `Hi {"fn` | free text; `{"fn` could become the trigger | 126: all but `"}` | yes |
| `Hi {"fn"` | the trigger has fired; both tags are candidates | `:` | no |
| `Hi {"fn": "` | begin, both tags | `n`, `s` | no |
| `Hi {"fn": "n` | begin, only `neg` | `e` | no |
| `Hi {"fn": "neg", "x": ` | content of `neg` | `-` and the ten digits | no |
| `Hi {"fn": "neg", "x": -4` | content of `neg` | the ten digits and `}` | no |
| `Hi {"fn": "neg", "x": -4}` | free text again | all 127 | yes |

- After `Hi {"fn` the output is still free text and may even end there. Almost every
  token is allowed, including `":` and `": `, which complete the trigger and continue
  correctly into both `begin` strings. The token `"}` is not: its quote fires the
  trigger and its brace then contradicts both tags, which need `:`.
- `Hi {"fa` would simply be free text: `{"fa` is not on the way to the trigger.
- After the trigger nothing but the two `begin` strings can follow. The pieces `nu`,
  `nul`, and `null` are not allowed after `"`: they start like `neg` and then
  differ.
- The number's automaton is accepting after `-4`, so `}` may end the tag; after `-`
  alone it may not.
- In the `say` tag, after `Hi {"fn": "say", "text": "hi` the allowed tokens are the
  letters, the space, the letter pieces (`tr`, `false`, `nul`, ...), the quote, and
  the token `"}`, which closes the string (content) and the tag (end) at once.

## Constraints

- At most 8 triggers and 8 tags; `vocab.size` is between 64 and 282.
- Every `end` is non-empty. Every `inner` comes from `CharDFA.from_regex`, so it is
  trimmed: a state that is not `DEAD` can always reach acceptance. No accepting
  state of an `inner` has a transition on the first byte of its tag's `end`, so
  "read on while the automaton can" never swallows an end string.
- Results are compared exactly, in every state the tests reach: the same ids in the
  same order, the same accepting flag.

## Why this matters

This is the mechanism behind tool calling with guaranteed arguments. XGrammar's
structural tags, exposed by vLLM and other servers as the `structural_tag` response
format, describe exactly this: free text, triggers such as `<function=`, and for
each tag a begin string, a schema for the content, and an end string. The engine
compiles it into one automaton and masks tokens with it, so the model may talk
freely and still cannot emit a call to a function that does not exist or arguments
that do not parse. The bugs are boundary bugs. Real triggers are several tokens
long, so a dispatcher that looks for the trigger inside one token never fires. A
tokenizer happily merges the last character of a trigger with whatever follows, so
tokens must be walked through every phase. And a server that lets generation end
between a trigger and the tag's end string returns half a tool call to a client
that was promised none or a whole one.
