## The idea

Special tokens live outside ordinary text. No sequence of user characters is supposed to produce them, because only the prompt-building code should insert them. However, prompts are often assembled as strings, so tokenizers offer a convenience that turns a special token's string name into its id. That convenience is safe for server-generated template strings and dangerous for user-supplied text. Since the tokenizer cannot tell who wrote the string, the caller must specify what is permitted via `allowed`.

`encode_with_specials` acts as a splitter in front of `encode_ordinary`. It cuts the text at special strings, emits their ids directly, and passes each intervening slice to `encode_ordinary`. Two design choices keep the behavior predictable:

- **Finding is separate from allowing.** The scan searches for every key in `specials` and only then checks whether the matched key is in `allowed`. If the scan ignored disallowed specials, they would silently fall through as ordinary text. Raising on disallowed specials turns a silent injection attempt into a visible error, while callers who want plain-text fallback can explicitly pass `"none_as_text"`.
- **Earliest position, then longest.** Scanning left to right and consuming each match makes the output independent of dictionary order. Choosing the longest key at a given index ensures that a longer special extending a shorter one (like `<|sep|>\n` extending `<|sep|>`) remains reachable.

Each ordinary run between specials must reach `encode_ordinary` whole and unmodified. BPE depends on adjacent characters, so `true` may encode as one token while `t`, `r`, `u`, `e` encoded separately become four. Whitespace is also meaningful, such as the newline after `<|im_start|>user` in a chat template.

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

- **Matching the shortest key first.** Checking keys in an order that matches `<|sep|>` before `<|sep|>\n` makes the longer token unreachable and turns the leftover `\n` into an ordinary token.
- **Always parsing specials.** Ignoring `allowed` introduces the prompt-injection bug directly. It still passes any test where all specials are allowed.
- **Encoding disallowed specials as text.** A caller who wants plain-text fallback passes `"none_as_text"`. When the caller passes a set, any other special must raise `ValueError`.
- **Splitting on one special at a time.** Calling `text.split(special)` in a loop over `specials` lets dictionary order decide conflicts, so an earlier dictionary entry can win over a match that starts earlier in `text` or is longer.
- **Searching from the right.** Using `rfind` or splitting recursively at the last match breaks on overlapping occurrences. For example, `"\n\n"` in `"a\n\n\nb"` must match at index 1, not index 2.
- **Dropping the tail.** When the loop reaches the end of `text`, the final ordinary run has not been encoded yet. A string with no specials at all is entirely tail.
- **Stripping or re-chunking ordinary runs.** Calling `.strip()` on runs or encoding character by character changes the token ids while keeping the decoded text almost identical, making the bug hard to spot by eye.

## Complexity

With $n$ characters in `text`, $k$ keys in `specials`, and special strings of length at most $m$, the scan takes $O(n \cdot k \cdot m)$ time in the worst case. Each ordinary character is passed to `encode_ordinary` at most once. Production tokenizers compile the special strings into a single automaton or regular expression to find all matches in $O(n)$ time.

## In the wild

This strict contract comes from `tiktoken`. Its `encode(text, allowed_special=set(), disallowed_special="all")` raises `ValueError` on any disallowed special token. Passing `disallowed_special=()` turns that check off and encodes special strings as ordinary text, while `encode_ordinary` skips special-token matching altogether. Two details differ from this problem. First, `tiktoken` scans the raw text for disallowed specials *before* splitting, so a disallowed special raises even if an allowed match would have overlapped it. Second, it matches via a regex alternation rather than longest-match.

Hugging Face tokenizers parse special tokens by default and provide `split_special_tokens=True` to encode them as text. Similarly, `llama.cpp`'s `llama_tokenize` takes a `parse_special` flag. Well-built servers tokenize template markers with specials enabled and user message content with specials disabled.
