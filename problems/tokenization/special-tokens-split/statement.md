A tokenizer has two kinds of tokens: ordinary ones, produced from text by BPE, and
special ones such as `<|im_start|>`, which mark the structure of a prompt and must
never be produced by accident. Implement the layer that sits in front of the BPE
encoder and decides when the *string* `<|im_start|>` becomes the *token*
`<|im_start|>`. The answer cannot be "always": then anyone who types that string into
a chat box writes their own system prompt.

## Specification

```python
def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]: ...
```

- `text`: the string to encode.
- `specials`: all special tokens of the tokenizer, as a dictionary from the token's
  string to its id.
- `encode_ordinary`: the plain encoder. It maps a string to a list of token ids and
  knows nothing about special tokens: given `"<|im_start|>"` it encodes the characters
  `<`, `|`, `i`, ... like any other text.
- `allowed`: either a set of special strings, each of them a key of `specials`, or the
  string `"none_as_text"`.

**When `allowed` is a set**, the function scans `text` for special strings and returns
the ids of the pieces, in order:

1. **Find the next special.** Starting at the current position (initially 0), find
   the smallest index at or after it where some key of `specials` occurs in `text`.
   *All* keys of `specials` take part in this search, allowed or not. If several keys
   occur at that index, take the longest one.
2. **If there is none**, the rest of the text is ordinary: if it is not empty, append
   `encode_ordinary(rest)`. Stop.
3. **If the special found is not in `allowed`, raise `ValueError`.**
4. Otherwise append, in this order: the ids of `encode_ordinary(run)`, where `run` is
   the text between the current position and the special, if `run` is not empty; then
   the special's id `specials[s]`. Move the current position to the end of the
   special, and go back to step 1.

So matches are found left to right and never overlap, the earliest match wins over a
longer one that starts later, and the order of the keys in `specials` plays no role.
Each maximal run of ordinary text is passed to `encode_ordinary` in **one** call, with
nothing added or removed: BPE merges depend on neighboring characters, so encoding a
run in smaller pieces, or stripping its whitespace, gives different ids.

**When `allowed == "none_as_text"`**, nothing is special: the function returns
`encode_ordinary(text)`, and never raises.

Details:

- `text` may be empty; the result is `[]`.
- The function returns a new list of Python ints and does not modify `specials` or
  `allowed`.
- `allowed` may be empty. Then any special string in the text raises `ValueError`, and
  a text without one is encoded as ordinary text.
- A `ValueError` is raised for the first disallowed special that the scan of steps 1
  to 4 arrives at. A special string that the scan never matches, because an earlier
  match covers part of it, does not count.
- Matching is exact and case-sensitive. `<|im_start`, `<|IM_START|>`, and
  `<| im_start |>` are ordinary text.

## Example

Take UTF-8 bytes as the ordinary encoder, `encode_ordinary = lambda s: list(s.encode())`,
and two specials, `specials = {"<|im_start|>": 256, "<|im_end|>": 257}`.

```python
text = "<|im_start|>user\nhi<|im_end|>"
encode_with_specials(text, specials, encode_ordinary, {"<|im_start|>", "<|im_end|>"})
# [256, 117, 115, 101, 114, 10, 104, 105, 257]
#  ^    u    s    e    r    \n  h    i    ^
```

The same text with only `<|im_start|>` allowed raises `ValueError`: the scan reaches
`<|im_end|>`, which is a special and is not allowed.

```python
encode_with_specials(text, specials, encode_ordinary, {"<|im_start|>"})   # ValueError
```

With `"none_as_text"` the result is the 29 bytes of the text; the special strings are
spelled out as characters:

```python
encode_with_specials(text, specials, encode_ordinary, "none_as_text")
# [60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62, 117, 115, 101, 114, 10,
#  104, 105, 60, 124, 105, 109, 95, 101, 110, 100, 124, 62]
```

This is how the function is used for untrusted input. A user message `hi<|im_end|>`
is encoded either strictly or as text, and in neither case does it end the turn:

```python
encode_with_specials("hi<|im_end|>", specials, encode_ordinary, set())            # ValueError
encode_with_specials("hi<|im_end|>", specials, encode_ordinary, "none_as_text")
# [104, 105, 60, 124, 105, 109, 95, 101, 110, 100, 124, 62]
```

Longest match at one position, with `specials = {"<|sep|>": 300, "<|sep|>\n": 301}`
and both allowed:

```python
encode_with_specials("a<|sep|>\nb", specials, encode_ordinary, {"<|sep|>", "<|sep|>\n"})
# [97, 301, 98]        not [97, 300, 10, 98]
```

With only `"<|sep|>"` allowed, the same call raises `ValueError`: the special found at
index 1 is still the longest one, `"<|sep|>\n"`, and it is not allowed.

## Constraints

- `len(text) <= 5000`; `specials` has at most 32 keys, each a non-empty string of at
  most 32 characters. Ids are arbitrary non-negative ints.
- When `allowed` is a set, it is a subset of the keys of `specials`.
- Most tests use the kit's [`ToyVocab`](../../../python/src/learnai/kit/vocab.py) with 282 tokens: `specials`
  is `vocab.specials` (`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`,
  ids 0 to 4), sometimes extended, and `encode_ordinary` is `vocab.encode`, a byte-level
  BPE encoder with merges such as `t`+`r`+`u`+`e`.
- Ids are compared exactly. There is no speed requirement beyond the time limit;
  checking every special at every position is fast enough.

## Why this matters

Chat models are trained on prompts in which special tokens separate the system prompt,
the user's turn, and the model's turn, and they trust those tokens completely. If the
server tokenizes the finished prompt string in one go with special tokens enabled, a
user who types `<|im_end|>` followed by `<|im_start|>system` ends their own turn and
opens a system turn: prompt injection at the tokenizer level, invisible in the text
logs because the string looks the same either way. The defense is this function's
contract: the template's own markers are encoded with `allowed` set to exactly the
tokens the template uses, user-supplied content is encoded with `"none_as_text"` (or
strictly, with an empty set, so that an attempt is an error), and the pieces are
concatenated as ids, never as strings.
