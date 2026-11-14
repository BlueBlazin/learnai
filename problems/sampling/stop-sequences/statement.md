Stop generation when the output contains a stop string, while the text is streamed to
the user piece by piece. A stop string rarely arrives as one token: it is spread over
several pieces, so the filter must hold back text that *might* be the beginning of a
stop string, release it when it turns out not to be, and never show any part of a
stop string that completes.

## Specification

```python
def stop_step(state, piece, stops): ...     # returns (state, emit, stopped)
def stop_flush(state): ...                  # returns a str
```

A stream is processed by calling `stop_step` once per piece, in order. The first call
gets `state = None`; each later call gets the state the previous call returned. The
state is yours: any Python value. `stops` is the same list on every call.

**The text and its cut.** Let `text` be the concatenation of all pieces so far. A
*match* is an occurrence of a stop string in `text`: a stop `s` and a start index `i`
with `text[i : i + len(s)] == s`. Its *end* is `i + len(s)`.

Generation stops at the first character that completes a match. So among all
matches, the one that counts is the one with the **smallest end**; if several matches
end at the same character, the one with the **smallest start** (the longest stop
string). With that match starting at `i`, the visible output of the stream is
`text[:i]`: everything before the stop string, nothing of it, nothing after it.

**`stop_step(state, piece, stops)`** returns a tuple `(state, emit, stopped)`:

- If `text` now contains a match, `stopped` is `True` and `emit` is the part of
  `text[:i]` that earlier calls have not emitted yet (possibly empty). The stream is
  over: neither function is called again for it.
- Otherwise `stopped` is `False`. Let `h` be the length of the longest suffix of
  `text` that is a **proper prefix** of some stop string (a prefix shorter than the
  stop string itself; `h = 0` if there is none). `emit` is the part of
  `text[: len(text) - h]` that earlier calls have not emitted yet. The last `h`
  characters are held back.

So after every call that does not stop, exactly the text that can still become part
of a stop string is held back: not a character more (ordinary text must reach the
user immediately) and not a character less (text that was shown cannot be
retracted).

**`stop_flush(state)`** is called once if the stream ends without a stop (the model
emitted EOS or ran out of budget), with the state of the last `stop_step` call, or
`None` if there was no call. It returns the held-back text, which is ordinary output
after all.

Parameters:

- `piece`: a `str`, the text of the newest token or tokens. It may be empty (special
  tokens and incomplete UTF-8 sequences detokenize to nothing); an empty piece emits
  nothing and changes nothing. It may also be long, and contain a whole stop string,
  or several.
- `stops`: a list of non-empty `str`. It may be empty (then everything is emitted at
  once and nothing ever stops), and its strings may overlap or be prefixes of each
  other. The order of the list has no influence on the result.
- `emit` is a `str`, `stopped` a `bool`.

The concatenation of every `emit`, plus the flushed text if the stream ended without
a stop, is the visible output defined above, **whatever way the text was split into
pieces**.

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

The newline of the second piece could begin `"\nUser:"`, so it is held; `"Us"` extends
the partial match. `"e #"` breaks it: `"\nUse "` is released, and the trailing `"#"`
is held because it could begin `"###"`. `"#1"` breaks that one as well. The last two
pieces complete `"\nUser:"`; nothing of it is emitted, and the piece `" next"` that
the model would have produced afterwards is never processed. The emitted text is
`"Sure.\nUse ##1."`.

If the stream had ended after `".\nUser"` instead, `stop_flush` would return
`"\nUser"`.

A stop inside a stop. With `stops = ["abcd", "bc"]` and the text `"xabcd"`, the match
`"bc"` ends at character 4 and `"abcd"` at character 5, so `"bc"` completes first and
the output is `"xa"`. That holds whether the text arrives as `["xabcd"]` (one call:
`emit = "xa"`, stopped) or as `["xab", "cd"]` (`emit = "x"`, holding `"ab"`; then
`emit = "a"`, stopped).

## Constraints

- At most 8 stop strings of at most 32 characters each; a stream has at most a few
  thousand pieces. Strings are Python `str`: lengths and indices count characters.
- `emit`, `stopped`, and the flushed text are compared exactly, at every step, with
  the definition above.
- No cost limit is judged, but nothing requires keeping the whole text: the state
  never needs more than the held-back characters.

## Why this matters

Every chat API has a `stop` parameter, and every chat template ends a turn with a
marker that some model, sooner or later, spells out as ordinary text instead of
emitting the special token. Stop strings are matched on text, not on tokens, because
the same characters can be tokenized in many ways: `"\nUser:"` may be one token or
four, and may begin in the middle of a token. A server that checks each token's text
alone misses most stops. One that streams first and matches later shows the user
`"\nUs"` and then goes silent, with half a stop string on the screen. And one that
solves this by delaying everything by the length of the longest stop string makes
every response stutter. The few lines you write here sit between the detokenizer and
the network in every streaming server.
