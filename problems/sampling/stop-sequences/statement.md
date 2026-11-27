Filter a streaming text output so that generation stops as soon as any stop string completes. Because a stop string rarely arrives as a single token, the filter must hold back trailing text that could begin a stop string, release that text if the match fails, and never emit any part of a completed stop string.

## Specification

```python
def stop_step(state, piece, stops): ...     # returns (state, emit, stopped)
def stop_flush(state): ...                  # returns a str
```

A caller processes a stream by invoking `stop_step` once per incoming `piece`, in order. The first call passes `state = None`, and each subsequent call passes the `state` returned by the previous call. You may use any Python value for `state`. The caller passes the same `stops` list on every call.

**Text and match selection.** Let `text` be the concatenation of all pieces received so far. A *match* is an occurrence of a stop string `s` in `text` at start index `i`, meaning `text[i : i + len(s)] == s`. Its *end* is `i + len(s)`.

Generation stops at the first character that completes any match. Among all matches in `text`, the winning match is the one with the **smallest end**. If multiple matches end at the same character, the winning match is the one with the **smallest start** (the longest stop string). If the winning match starts at `i`, the visible output of the stream is `text[:i]`. Nothing from the stop string or after it is emitted.

**`stop_step(state, piece, stops)`** returns `(state, emit, stopped)`:

- If `text` contains a match, `stopped` is `True` and `emit` is the unemitted portion of `text[:i]` (which may be empty). The stream then ends, and neither function is called again for it.
- Otherwise, `stopped` is `False`. Let `h` be the length of the longest suffix of `text` that is a **proper prefix** of some stop string (strictly shorter than that stop string, or `0` if none exists). `emit` is the unemitted portion of `text[: len(text) - h]`, and the last `h` characters remain held back.

After every non-stopping call, the filter holds back only the characters that can still become part of a stop string. Holding fewer risks emitting text that cannot be retracted, while holding more delays ordinary output unnecessarily.

**`stop_flush(state)`** is called once if the stream finishes without hitting a stop string (for example, after EOS or a token limit). It receives the `state` from the last `stop_step` call, or `None` if `stop_step` was never called, and returns the held-back text.

Parameters and return types:

- `piece`: a `str` containing the text of the newest token or tokens. It may be empty (as with special tokens or incomplete UTF-8 sequences), in which case it emits nothing and leaves the state unchanged. It may also be long and contain one or more complete stop strings.
- `stops`: a list of non-empty `str`. It may be empty, in which case all text is emitted immediately and nothing stops. Stop strings may overlap or be prefixes of each other, and their order in the list does not affect the result.
- `emit`: a `str`.
- `stopped`: a `bool`.

Concatenating every `emit` (plus the return value of `stop_flush` if the stream did not stop) must equal the visible output defined above, **regardless of how `text` was split into pieces**.

## Example

```python
stops = ["\nUser:", "###"]
```

| piece | `emit` | held back afterwards | `stopped` |
|---|---|---|---|
| `"Sure"` | `"Sure"` | `""` | `False` |
| `".\n"` | `"."` | `"\n"` | `False` |
| `"Us"` | `""` | `"\nUs"` | `False` |
| `"e #"` | `"\nUse "` | `"#"` | `False` |
| `"#1"` | `"##1"` | `""` | `False` |
| `".\nUser"` | `"."` | `"\nUser"` | `False` |
| `":"` | `""` | | `True` |

In piece 2, the newline can start `"\nUser:"`, so it is held back. Piece 3 (`"Us"`) extends that partial match. Piece 4 (`"e #"`) breaks it, so `"\nUse "` is released while the trailing `"#"` is held as a prefix of `"###"`. Piece 5 (`"#1"`) breaks that partial match too. Pieces 6 and 7 complete `"\nUser:"`, so nothing from it is emitted and any subsequent piece such as `" next"` is never processed. The total emitted text is `"Sure.\nUse ##1."`.

![The stream after pieces 2, 3, 4, and 7 of the table. Filled cells are held back, and the stop string is never emitted.](assets/stream-steps.svg)

If the stream had ended after `".\nUser"` instead, `stop_flush` would return `"\nUser"`.

**A stop inside a stop.** Consider `stops = ["abcd", "bc"]` and the text `"xabcd"`. The match `"bc"` ends at character 4 while `"abcd"` ends at character 5, so `"bc"` completes first and the visible output is `"xa"`. This holds whether the text arrives as `["xabcd"]` (one call returning `emit = "xa"`, `stopped = True`) or as `["xab", "cd"]` (first `emit = "x"` while holding `"ab"`, then `emit = "a"` with `stopped = True`).

## Constraints

- At most 8 stop strings of at most 32 characters each, and at most a few thousand pieces per stream. Strings are Python `str`, so lengths and indices count characters.
- `emit`, `stopped`, and the flushed text are compared exactly at every step against the specification.
- No memory or runtime limit is judged, though the state never needs to store more than the held-back characters.

## Why this matters

Every chat API exposes a `stop` parameter, and chat templates end turns with markers that models sometimes emit as plain text instead of special tokens. Stop strings must be matched on decoded text rather than token IDs because the same string can be tokenized in many ways. For example, `"\nUser:"` may span one token or four, and it may start mid-token.

Checking each token in isolation misses split stop strings. Streaming first and checking later leaves partial markers like `"\nUs"` stuck on the user's screen. Delaying every response by the maximum stop-string length avoids leaks but makes output stutter. This small filter sits between the detokenizer and the network in every streaming inference server.
