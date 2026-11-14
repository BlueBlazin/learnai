## The idea

Split the generated text into three parts at any moment:

```
text = [ emitted | held | not generated yet ]
```

*Emitted* text is on the user's screen and cannot change. *Held* text has been
generated but is kept back. The whole problem is choosing the boundary between the
two, and the right choice follows from one question: **which already-generated
characters can still end up inside a stop string?**

A stop string that has not completed yet must end in text that does not exist yet.
So it either lies entirely in the future, or it straddles the present: it starts at
some position of the current text and runs past its end. In the second case the
current text *ends with a proper prefix of that stop string*. Conversely, any suffix
of the text that is a proper prefix of a stop string can still be completed. The
characters at risk are therefore exactly the longest such suffix, and that is what
must be held:

- holding less shows the user the beginning of a stop string that may complete;
- holding more delays text that is certainly ordinary output.

This choice also makes matching local. Every match that completes in the new piece
starts either in the piece or in the held text, so `held + piece` is the only string
that ever has to be searched, and the state is just the held text, which is shorter
than the longest stop string.

**Which match?** Generation stops at the first character that completes any stop
string, so the match that counts is the one that *ends* first. When a piece contains
several matches, picking the earliest end gives the result that feeding the same
text one character at a time would give; that is what makes the output independent
of the chunking. Matches that end at the same character differ only in where they
start, and the cut goes before the longer one, since all of it is stop string.

## Reference solution (Python)

```python
def stop_step(state, piece, stops):
    buffer = (state or "") + piece

    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[: best[1]], True

    hold = 0
    for stop in stops:
        for n in range(min(len(stop) - 1, len(buffer)), hold, -1):
            if buffer.endswith(stop[:n]):
                hold = n
                break
    cut = len(buffer) - hold
    return buffer[cut:], buffer[:cut], False


def stop_flush(state):
    return state or ""
```

Why the cut can never fall inside text that was already emitted: a match found in
`buffer` starts in `buffer`, and `buffer` consists of unemitted text only.

Why the held text never hides a complete match: it is a *proper* prefix of a stop
string and was searched, as part of `buffer`, before it was held.

## Common mistakes

- **Checking each piece alone.** `if stop in piece` finds a stop string only when one
  token contains all of it. With real tokenizers that is the exception.
- **Emitting first, matching later.** Keeping the recent text only for detection
  finds the stop, but its first characters are already on the screen. The output
  ends in `"\nUs"`.
- **A fixed hold-back window.** Always holding the last `max(len(stop)) - 1`
  characters is correct in total and simple, and it is what some engines do (see
  below). It delays every piece of ordinary text by that many characters, which a
  user sees as stutter when a stop string is long.
- **Restarting after a failed partial match.** When `held + piece` stops being a
  prefix of a stop string, the held text is not necessarily free to go: with the stop
  `"aab"` and the text `"aaa"`, the last two characters are a new partial match. This
  is the string-matching mistake that the KMP failure function exists for. Recomputing
  the longest suffix from the whole buffer avoids it.
- **Letting the list order decide.** Returning the first stop string *of the list*
  that occurs cuts too late when a later entry of the list occurs earlier in the
  text.
- **Preferring the match that starts first.** It sounds like the same thing as
  "ends first" and differs only when one stop string lies inside a longer match, as
  `"bc"` in `"abcd"`. Then the answer depends on whether the text arrived in one
  piece or two, which no user of an API can predict.
- **Forgetting the flush.** A response that ends in `"#"` or in a newline loses its
  last characters when a stop string begins with them.

## Complexity

With $S$ stop strings of length at most $L$ and a piece of length $n$, the buffer has
at most $L - 1 + n$ characters. The search is $O(S \cdot L \cdot (L + n))$ in the
worst case with naive matching and the suffix test $O(S \cdot L^2)$; both are
independent of the length of the text generated so far. The state is at most
$L - 1$ characters. An Aho-Corasick automaton over the stop strings does both jobs
in $O(n)$ per piece with the automaton state as the only state: its depth is the
length to hold back. With a handful of short stop strings the simple version is
faster in practice.

## In the wild

- **llama.cpp** (`tools/server/server-context.cpp`) searches the unsent text for
  every stop word, takes the smallest start position among the full matches, and
  otherwise looks for a partial match at the end of the text; while there is one it
  sends nothing. It is the approach of this problem, with "earliest start" where
  this problem says "earliest end".
- **vLLM** (`vllm/v1/engine/detokenizer.py`) holds back a fixed
  `max(len(s) for s in stop) - 1` characters while streaming, and
  `check_stop_strings` selects "the stop string that completes earliest in the
  text", so that the result matches appending one token at a time: the rule used
  here, motivated by speculative decoding delivering several tokens in one step.
- **Hugging Face transformers** has `StopStringCriteria`, which works on token ids:
  it precomputes, for every token, how it can overlap each stop string, so the check
  runs on the GPU without detokenizing. It stops generation but does not trim the
  output.
- The OpenAI-style API contract (`stop`: "the returned text will not contain the
  stop sequence") is what all of them implement.

Stop strings are matched on characters, so they compose with the streaming
detokenizer of `streaming-detokenizer`: bytes become complete characters first, and
pieces of characters then go through this filter.
