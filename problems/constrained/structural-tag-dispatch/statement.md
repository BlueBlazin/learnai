An assistant with tool calling writes ordinary prose most of the time and emits a structured tool call only occasionally. A structural tag dispatcher enforces this mixed format as a token constraint. Output stays unconstrained until the model emits a *trigger* string, then must complete one of the tags that start with that trigger (its begin string, content matching its language, and end string), and then returns to unconstrained prose. Because triggers, tag boundaries, and vocabulary tokens do not align, the constraint must track state across byte boundaries.

![This problem is Allowed tokens and Advance state, free outside a tag and constrained inside one.](assets/map.svg)

## Specification

```python
class TagDispatch:
    def __init__(self, triggers, tags, vocab): ...
    def start(self): ...  # the state before any token
    def allowed(self, state) -> list[int]: ...
    def advance(self, state, token): ...  # the state after `token`
    def is_accepting(self, state) -> bool: ...
```

This class follows the `TokenConstraint` protocol from `constrained-greedy-decode`.

- `triggers`: a list of non-empty strings.
- `tags`: a list of tuples `(begin, inner, end)`. `begin` and `end` are strings, and `inner` is a [`CharDFA`](../../../python/src/learnai/kit/dfa.py) built with `CharDFA.from_regex` for the tag's content language. A tag *belongs to* the trigger that its `begin` string starts with.
- `vocab`: a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py). Calling `vocab.bytes(t)` returns the byte string of token `t`, which is empty for special tokens.

All matching operates on UTF-8 bytes. Every input string refers to its UTF-8 encoding, and the generated output is the concatenation of the generated tokens' byte strings.

### The language

Read the output bytes from left to right. At any point, the reader is either outside tags (in *free text*) or inside a tag.

1. **Free text.** Any byte is valid. The reader watches for the first position where the current stretch of free text (read since the start of output, or since the last tag ended) ends with a trigger. Bytes from before the current stretch of free text never count toward a trigger. At that position, the trigger *fires*.
2. **Begin.** Once a trigger fires, the following bytes must extend it to spell the `begin` string of one of the trigger's tags. All tags whose `begin` still matches the bytes read so far remain candidates. A byte that no candidate expects next is invalid. A trigger with no belonging tags has no candidates, so it can never be completed and acts as a forbidden string in free text.
3. **Content.** Once a tag's `begin` is complete, subsequent bytes are fed to that tag's `inner` automaton starting from `inner.start`. Calling `inner.step(q, byte)` from state `q` returns the next state, or `DEAD` (`-1`) if no transition exists. Content continues as long as `inner` can transition on the next byte.
4. **End.** When `inner` cannot transition on the next byte, `inner` must be in an accepting state (`inner.is_accepting(q)`) and that byte must be the first byte of the tag's `end` string. Otherwise, the byte is invalid. The remaining bytes of `end` must follow in exact order. After the last byte of `end`, the reader returns to free text and starts a fresh stretch (so bytes from `end` do not count toward a future trigger).

An output is **valid so far** if every byte in it is valid. It is **complete** if it is valid so far and the reader is currently in free text (outside every tag). Free text may stop anywhere, including after a partial trigger, because a partial trigger has not fired.

### The methods

Let `text(state)` be the concatenated bytes of all tokens passed to `advance` since `start()`.

- **`allowed(state)`** returns, in increasing order, the IDs of all tokens `t` such that `vocab.bytes(t)` is non-empty and `text(state) + vocab.bytes(t)` is valid so far. All bytes of a token are processed in order, even across phase boundaries. For example, one token may finish a trigger and begin a tag's `begin` string, or finish the content and include the entire `end` string. Special tokens (empty byte strings) are never included.
- **`advance(state, token)`** returns the state for `text(state) + vocab.bytes(token)`. It raises `ValueError` if `token` is not allowed in `state`.
- **`is_accepting(state)`** returns whether `text(state)` is complete.

States must be opaque to the caller, hashable, and immutable.

The constructor raises `ValueError` if the configuration is ambiguous:

- One trigger is a substring of another trigger (including prefixes and duplicates). In that case, the shorter trigger would always fire first, or both would fire at once.
- A tag's `begin` does not start with any trigger.
- One tag's `begin` is a prefix of (or equal to) another tag's `begin`. In that case, completing the shorter `begin` would leave the choice of tag ambiguous.

When these checks pass, every `begin` starts with exactly one trigger, and multiple tags may share the same trigger.

Interact with each `inner` automaton only through `start`, `step`, and `is_accepting`. The test oracles `accepts`, `accepts_prefix`, and `is_live` are forbidden and rejected by the harness.

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

The vocabulary contains 127 non-empty tokens, including every printable ASCII character and multi-character pieces such as `{"`, `":`, `": `, `"}`, `, `, `true`, and `nul`.

| text so far | where the reader is | allowed | accepting |
|---|---|---|---|
| (empty) | free text | all 127 | yes |
| `Hi {"fn` | free text (`{"fn` could become the trigger) | 126 (all except `"}`) | yes |
| `Hi {"fn"` | trigger fired, both tags are candidates | `:` | no |
| `Hi {"fn": "` | begin, both tags | `n`, `s` | no |
| `Hi {"fn": "n` | begin, only `neg` | `e` | no |
| `Hi {"fn": "neg", "x": ` | content of `neg` | `-` and the ten digits | no |
| `Hi {"fn": "neg", "x": -4` | content of `neg` | the ten digits and `}` | no |
| `Hi {"fn": "neg", "x": -4}` | free text again | all 127 | yes |

- After `Hi {"fn`, the reader is still in free text and may stop there. Almost every token is allowed, including `":` and `": `, which complete the trigger and continue validly into both `begin` strings. The token `"}` is disallowed because its quote fires the trigger and its closing brace contradicts both tags, which expect `:`.
- `Hi {"fa` stays in free text because `{"fa` is not a prefix of the trigger.
- After the trigger fires, only bytes matching one of the two `begin` strings can follow. Multi-character pieces like `nu`, `nul`, and `null` are disallowed after `"` because they start like `neg` and then diverge.
- The `neg` content automaton is accepting after `-4`, so `}` can end the tag. After `-` alone, the automaton is not accepting, so `}` is disallowed.
- In the `say` tag, after `Hi {"fn": "say", "text": "hi`, the allowed tokens are letters, space, multi-letter pieces (`tr`, `false`, `nul`, and so on), `"`, and `"}`. The token `"}` closes the quoted string (content) and the tag (`end`) in a single step.

## Constraints

- At most 8 triggers and 8 tags. `vocab.size` is between 64 and 282.
- Every `end` string is non-empty.
- Every `inner` automaton is built with `CharDFA.from_regex`, so it is trimmed (every non-`DEAD` state can reach an accepting state).
- No accepting state of an `inner` automaton has a transition on the first byte of its tag's `end` string. Reading while `inner` can transition therefore never swallows the start of `end`.
- Outputs are checked for exact equality in every state the tests reach (the same token IDs in the same order, and the same `is_accepting` boolean).

## Why this matters

Structural tags power constrained tool calling in production inference engines. XGrammar's `structural_tag` format (exposed by vLLM and similar servers) uses this exact structure of free text, triggers such as `<function=`, and tags with a begin string, a content grammar, and an end string. Compiling the whole specification into one automaton lets the model speak freely while guaranteeing that every emitted tool call names a real function with valid arguments.

Most failures in practice happen at token boundaries. Real triggers span multiple tokens, so checking for a trigger inside a single token misses it entirely. Tokenizers also merge the last byte of a trigger or content string with the byte that follows, so every token must be stepped byte by byte across phase transitions. Finally, allowing generation to stop between a trigger and its end string hands a half-written tool call to a client expecting either plain text or a complete call.
