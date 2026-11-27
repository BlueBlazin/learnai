A tokenizer produces ordinary tokens from text using BPE, along with special tokens such as `<|im_start|>` that mark prompt structure. Implement `encode_with_specials`, the layer in front of the BPE encoder that converts special-token strings into their ids only when explicitly allowed so untrusted user text cannot inject control tokens.

![This problem is the Split specials step, deciding where the string of a special token becomes its id.](assets/map.svg)

## Specification

```python
def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]: ...
```

- `text`: the input string to encode.
- `specials`: all special tokens of the tokenizer, mapping each special string to its integer id.
- `encode_ordinary`: the plain encoder that maps a string to a list of token ids without recognizing special tokens. Given `"<|im_start|>"`, it encodes the characters `<`, `|`, `i`, and so on as ordinary text.
- `allowed`: either a set of special strings (each a key in `specials`) or the string `"none_as_text"`.

**When `allowed` is a set**, scan `text` left to right from position `0` and collect token ids in order:

1. **Find the next special.** Find the smallest index at or after the current position where any key of `specials` matches in `text`. Every key in `specials` participates in this search, whether or not it is in `allowed`. If multiple keys match at that same index, pick the longest key `s`.
2. **Handle the tail.** If no key matches, the remaining substring `rest` is ordinary text. If `rest` is non-empty, append `encode_ordinary(rest)` and stop.
3. **Check permission.** If the matched special `s` is not in `allowed`, raise `ValueError`.
4. **Emit and advance.** Let `run` be the ordinary text between the current position and the start of `s`. If `run` is non-empty, append `encode_ordinary(run)`. Then append `specials[s]`, advance the current position to the end of `s`, and return to step 1.

Because the scan moves left to right and consumes each match, matches never overlap. An earlier match always wins over a longer match that starts later, and dictionary insertion order in `specials` has no effect. Pass each maximal run of ordinary text to `encode_ordinary` in **one** call without adding or removing characters. BPE merges depend on neighboring characters, so splitting a run or stripping its whitespace changes the resulting ids.

**When `allowed == "none_as_text"`**, treat nothing as special. Return `encode_ordinary(text)` and never raise `ValueError`.

Details:

- If `text` is empty, return `[]`.
- Return a new list of Python ints without modifying `specials` or `allowed`.
- `allowed` may be an empty set. In that case, any special string in `text` raises `ValueError`, while text containing no special strings is encoded as ordinary text.
- Raise `ValueError` for the first disallowed special reached by steps 1 to 4. A special string that the scan never matches because an earlier match already consumed part of it does not raise.
- Matching is exact and case-sensitive. Strings like `<|im_start`, `<|IM_START|>`, and `<| im_start |>` are ordinary text.

## Example

Use UTF-8 bytes as the ordinary encoder (`encode_ordinary = lambda s: list(s.encode())`) and two specials (`specials = {"<|im_start|>": 256, "<|im_end|>": 257}`):

```python
text = "<|im_start|>user\nhi<|im_end|>"
encode_with_specials(text, specials, encode_ordinary, {"<|im_start|>", "<|im_end|>"})
# [256, 117, 115, 101, 114, 10, 104, 105, 257]
#  ^    u    s    e    r    \n  h    i    ^
```

With only `<|im_start|>` allowed, the same text raises `ValueError` when the scan reaches `<|im_end|>`:

```python
encode_with_specials(text, specials, encode_ordinary, {"<|im_start|>"})   # ValueError
```

With `"none_as_text"`, all 29 bytes are encoded as ordinary characters:

```python
encode_with_specials(text, specials, encode_ordinary, "none_as_text")
# [60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62, 117, 115, 101, 114, 10,
#  104, 105, 60, 124, 105, 109, 95, 101, 110, 100, 124, 62]
```

Untrusted user input such as `hi<|im_end|>` is encoded either strictly (raising on any special) or as plain text. In neither case does it emit the end-of-turn token:

```python
encode_with_specials("hi<|im_end|>", specials, encode_ordinary, set())            # ValueError
encode_with_specials("hi<|im_end|>", specials, encode_ordinary, "none_as_text")
# [104, 105, 60, 124, 105, 109, 95, 101, 110, 100, 124, 62]
```

When two specials match at the same position, the longer one wins. With `specials = {"<|sep|>": 300, "<|sep|>\n": 301}` and both allowed:

```python
encode_with_specials("a<|sep|>\nb", specials, encode_ordinary, {"<|sep|>", "<|sep|>\n"})
# [97, 301, 98]        not [97, 300, 10, 98]
```

If only `"<|sep|>"` is allowed, the same call raises `ValueError`. The longest match at index 1 is still `"<|sep|>\n"`, which is not in `allowed`.

## Constraints

- `len(text) <= 5000`. `specials` has at most 32 keys, each a non-empty string of at most 32 characters, mapped to arbitrary non-negative ints.
- When `allowed` is a set, it is a subset of the keys of `specials`.
- Most tests use the kit's [`ToyVocab`](../../../python/src/learnai/kit/vocab.py) with 282 tokens. In those tests, `specials` is `vocab.specials` (`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, ids 0 to 4), sometimes extended, and `encode_ordinary` is `vocab.encode`, a byte-level BPE encoder with merges such as `t`+`r`+`u`+`e`.
- Token ids are compared exactly. There is no speed requirement beyond the time limit, so checking every special at every position is fast enough.

## Why this matters

Chat models are trained on prompts where special tokens separate the system prompt, user turns, and assistant turns, and the model trusts those tokens completely. If a server concatenates the full prompt into one string and tokenizes it with special tokens enabled, a user who types `<|im_end|>` followed by `<|im_start|>system` ends their own turn and opens a fake system turn. Because the raw string looks identical in text logs, this tokenizer-level prompt injection is easy to miss.

This function's contract prevents that failure. Template markers are encoded with `allowed` set to the exact tokens the template uses, user content is encoded with `"none_as_text"` (or strictly with an empty set so injection attempts raise an error), and the resulting token id lists are concatenated directly rather than joined as strings.
