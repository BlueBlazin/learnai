## The idea

Special tokens live outside the text. No sequence of characters is *supposed* to
produce them; they are inserted by the code that builds the prompt. But prompts are
built as strings, so tokenizers offer a convenience: write the special token's name in
the string, and the tokenizer turns it into the id. That convenience is safe for
strings the server wrote and dangerous for strings a user wrote, and the tokenizer
cannot tell which is which. So the caller has to say, and `allowed` is how.

The function is a splitter in front of the real encoder. It cuts the text at special
strings, emits their ids directly, and hands everything in between to
`encode_ordinary`. Two design decisions make it predictable:

- **Finding is separate from allowing.** The scan looks for every special, and only
  then asks whether the one it found is allowed. If the search only looked for allowed
  specials, a disallowed one would silently fall through as text; whether that is
  acceptable is exactly what the caller should decide, by choosing between a strict
  set (an error) and `"none_as_text"` (text). Raising by default turns a silent
  injection attempt into a visible failure.
- **Earliest position, then longest.** Scanning left to right and consuming each match
  makes the result independent of dictionary order. Preferring the longest key at a
  position means a token whose string extends another one (`<|sep|>` and `<|sep|>\n`)
  can be produced at all.

The runs between specials must reach `encode_ordinary` whole and unmodified. BPE is
context-dependent: `true` is one token, and `t`, `r`, `u`, `e` encoded separately are
four. Whitespace is content too: the newline after `<|im_start|>user` is part of the
format the model was trained on.

## Reference solution (Python)

```python
def encode_with_specials(text, specials, encode_ordinary, allowed):
    if allowed == "none_as_text":
        return encode_ordinary(text)

    by_length = sorted(specials, key=len, reverse=True)
    ids = []
    run_start = 0
    pos = 0
    while pos < len(text):
        special = next((s for s in by_length if text.startswith(s, pos)), None)
        if special is None:
            pos += 1
            continue
        if special not in allowed:
            raise ValueError(f"disallowed special token {special!r} at position {pos}")
        if pos > run_start:
            ids.extend(encode_ordinary(text[run_start:pos]))
        ids.append(specials[special])
        pos += len(special)
        run_start = pos
    if run_start < len(text):
        ids.extend(encode_ordinary(text[run_start:]))
    return ids
```

## Common mistakes

- **Shortest match.** Trying keys in an order that finds `<|sep|>` before
  `<|sep|>\n` leaves the longer token unreachable, and the leftover `\n` becomes an
  ordinary token the model did not see at that place in training.
- **Always parsing specials.** Ignoring `allowed` is the vulnerable default described
  above. It passes every test in which the caller allows everything.
- **Encoding a disallowed special as text.** Lenient, and wrong for this contract: the
  caller who wanted leniency asks for `"none_as_text"`; the caller who passed a set
  wants to hear about anything else.
- **Splitting on one special at a time.** `text.split(special)` in a loop over the
  dictionary resolves conflicts by dictionary order: a special listed first claims its
  occurrences even where another special starts earlier or is longer.
- **Searching from the right.** `rfind`, or a recursive split at the last match, gives
  the same result until two matches overlap; then the later one wins. `"\n\n"` in
  `"a\n\n\nb"` must match at index 1, not at index 2.
- **Dropping the tail.** The loop ends when the scan reaches the end of the text, and
  the last run has not been encoded yet. A text with no special at all is the extreme
  case: the whole text is the tail.
- **Stripping or re-chunking the runs.** `.strip()` on each run, or encoding character
  by character, changes the ids without changing the decoded text much, so it is easy
  to miss by eye.

## Complexity

With $n$ characters, $k$ specials, and special strings of length at most $m$, the scan
is $O(n \cdot k \cdot m)$ in the worst case and the calls to `encode_ordinary` cover each
character once. Production tokenizers compile the specials into one automaton or
regular expression and find all matches in a single pass, $O(n)$.

## In the wild

`tiktoken` is where the strict contract comes from. Its `encode(text,
allowed_special=set(), disallowed_special="all")` raises `ValueError` when the text
contains a special token that is not allowed; `disallowed_special=()` turns the check
off, so that special strings are encoded as ordinary text; and `encode_ordinary`
is the fast path that ignores specials entirely. Two differences from this problem:
tiktoken searches the raw text for disallowed specials *before* splitting, so a
disallowed special raises even where an allowed match would have covered part of it,
and it finds specials with a regular-expression alternation rather than by longest
match. Hugging Face tokenizers parse special tokens in the input by default; the
`split_special_tokens=True` argument is their "as text" mode. `llama.cpp`'s
`llama_tokenize` takes a `parse_special` flag for the same purpose. The pattern in a
well-built server is always the same: the chat template's markers are tokenized with
specials enabled, message content with specials disabled.
