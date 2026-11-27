## The idea

At any point during generation, the text splits into three regions:

```
text = [ emitted | held | not generated yet ]
```

*Emitted* text has already reached the user and cannot be retracted. *Held* text has been generated but not sent yet. Choosing the boundary between them comes down to one question: **which already-generated characters can still end up inside a stop string?**

Any stop string that has not completed yet must end in future text. It either lies entirely in the future, or it starts in the current text and extends past its end. In the second case, the current text ends with a *proper prefix* of that stop string. Conversely, any suffix of the current text that is a proper prefix of some stop string can still complete into a match. The characters at risk are therefore exactly the longest such suffix.

- Holding fewer characters risks showing the start of a stop string that later completes.
- Holding more characters delays text that is guaranteed to be ordinary output.

This boundary also keeps matching local. Any match that completes during a new `piece` must start inside either `held` or `piece`. As a result, `held + piece` is the only string we ever need to search, and `state` only needs to store `held`, whose length is always strictly less than the longest stop string.

**Choosing the winning match.** Generation stops at the first character that completes any stop string, so the winning match is the one that *ends* earliest. Picking the earliest end when a piece contains multiple matches produces the same result as feeding the text one character at a time, which guarantees chunk invariance. When two matches end at the same character, cutting at the earlier start removes the longer stop string completely.

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

A match found in `buffer` can never cut into already-emitted text because `buffer` contains only unemitted characters. Likewise, the held suffix can never hide a complete match because it is a proper prefix of a stop string and was already searched as part of `buffer` before being held.

## Common mistakes

- **Checking each piece alone.** Testing `stop in piece` only catches a stop string when a single token contains all of it, which rarely happens with real tokenizers.
- **Emitting first and matching later.** Keeping recent text only for detection catches the stop, but its prefix (such as `"\nUs"`) has already been sent to the user.
- **Using a fixed hold-back window.** Always holding the last `max(len(s) for s in stops) - 1` characters is simple and produces the right final string, but it delays every piece of ordinary text by that many characters and causes visible stutter when stop strings are long.
- **Resetting completely after a failed partial match.** When `held + piece` stops matching a stop prefix, the held characters are not always safe to emit. With stop `"aab"` and text `"aaa"`, the last two characters form a new partial match. Recomputing the longest suffix over the full buffer avoids this bug (the same overlap issue addressed by the KMP failure function).
- **Letting the `stops` list order decide.** Returning the first stop string in the list that appears in `buffer` cuts too late when a later list entry completes earlier in the text.
- **Preferring the match that starts first.** Choosing the smallest start instead of the smallest end breaks chunk invariance when one stop lies inside another, such as `"bc"` inside `"abcd"`. The output then depends on how the text was split into pieces.
- **Forgetting the flush.** If a response ends with `"#"` or a newline and a stop string starts with that character, skipping `stop_flush` drops the final characters.

## Complexity

Let $S$ be the number of stop strings, $L$ be the maximum stop-string length, and $n$ be the length of `piece`. The buffer has at most $L - 1 + n$ characters. Naive substring search takes $O(S \cdot L \cdot (L + n))$ time in the worst case, and the suffix check takes $O(S \cdot L^2)$ time. Both are independent of the total generated length, and the state stores at most $L - 1$ characters. An Aho-Corasick automaton over the stop strings handles both matching and suffix tracking in $O(n)$ time per piece using only the current automaton node as state, where the node depth is the hold-back length. For a small list of short stop strings, the simple implementation above is faster in practice.

## In the wild

- **llama.cpp** (`tools/server/server-context.cpp`) searches unsent text for every stop word, picks the smallest start position among full matches, and otherwise holds back output while a partial suffix match exists. It uses the same strategy as this problem, except with earliest-start tie-breaking instead of earliest-end.
- **vLLM** (`vllm/v1/engine/detokenizer.py`) holds back a fixed `max(len(s) for s in stop) - 1` characters while streaming, and `check_stop_strings` selects the stop string that completes earliest so that multi-token steps from speculative decoding match single-token streaming.
- **Hugging Face transformers** provides `StopStringCriteria`, which precomputes how each token in the vocabulary can overlap each stop string so the check runs on token IDs without detokenizing. It stops generation but does not trim the output.
- All of these implement the OpenAI-style `stop` API contract, which guarantees that the returned text never contains the stop sequence.

Because stop strings are matched on characters, this filter composes directly with `streaming-detokenizer`. Raw bytes become complete Unicode characters first, and those character pieces then pass through `stop_step`.
