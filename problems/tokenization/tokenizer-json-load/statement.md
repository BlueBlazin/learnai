Build a working tokenizer from the `tokenizer.json` file of a real model, IBM's Granite 4.0 350M, and reproduce the token ids of the `tokenizers` library exactly. You will reuse the BPE merge loop from `bpe-encode` and wrap it in the surrounding stages that a real model file specifies: added tokens, regex pre-tokenization, a byte-level alphabet, and vocabulary id lookups.

![This problem is the Split specials, Encode, and Detokenize steps, built from the file of a real model.](assets/map.svg)

## Specification

```python
def load_tokenizer(
    tokenizer_json: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]: ...
```

`tokenizer_json` is the parsed JSON dictionary. `load_tokenizer` returns a pair of functions, `(encode, decode)`. Use these four fields from `tokenizer_json`:

| path in the JSON | content |
|---|---|
| `["model"]["vocab"]` | dict mapping each token string to its integer id |
| `["model"]["merges"]` | list of `[left, right]` pairs of token strings, where the list index is the pair's **rank** |
| `["added_tokens"]` | list of objects with `"content"` (the token's text), `"id"`, and `"special"` (a bool) |
| `["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]` | regular expression that splits text into words |

The remaining fields are identical across all test files and need no special handling. There is no normalizer and no post-processor. The second pre-tokenizer is always `ByteLevel` with `add_prefix_space` set to `false`, and every added token has `lstrip`, `rstrip`, `single_word`, and `normalized` set to `false`.

**`encode(text)`** returns the list of token ids produced by these four steps:

1. **Added tokens.** Scan `text` from left to right for the `content` of any added token, **whether or not `special` is `true`**. The leftmost match wins. If multiple added tokens start at the same position, the longest match wins. Emit each match's `id` and resume scanning immediately after it. Encode every non-empty stretch of text before, between, or after matches independently using steps 2 to 4.
2. **Words.** Split each stretch into words using the regex from the file. The words are its successive non-overlapping matches from left to right, as returned by `re.finditer`. Merges never cross a word boundary. Because the pattern uses `\p{L}` and `\p{N}`, which Python's `re` module does not support, compile it with `re.compile(translate_pattern(pattern))`, where `translate_pattern` is imported from `learnai.kit.bpe`. Always read the pattern from `tokenizer_json`.
3. **Bytes as characters.** Encode each word as UTF-8 and map each byte to one character of the ByteLevel alphabet, which is how `vocab` represents tokens:
   - Map the 188 bytes `0x21`–`0x7E`, `0xA1`–`0xAC`, and `0xAE`–`0xFF` to the character with the same code point (`chr(byte)`).
   - Map the remaining 68 bytes, in increasing byte order, to consecutive code points starting at U+0100. For example, `0x00` maps to `Ā` (U+0100), newline `0x0A` maps to `Ċ` (U+010A), space `0x20` maps to `Ġ` (U+0120), and `0x7F` maps to `ġ` (U+0121).
4. **BPE.** Start with the word as a list of single-character strings. Repeatedly find the adjacent pair in the list that appears in `merges` with the lowest rank, and replace every non-overlapping occurrence of that pair from left to right with the concatenation of its two strings. Stop when no adjacent pair appears in `merges`. Look up `vocab[s]` for each remaining string `s` to get the word's token ids.

Do not add extra tokens such as beginning-of-text markers. `encode("")` returns `[]`.

**`decode(ids)`** converts a list of token ids back into a string. Collect the bytes of each token in order, concatenate them, and decode the full byte sequence as UTF-8 with invalid sequences replaced by U+FFFD (`errors="replace"`). For an added token, its bytes are its `content` encoded as UTF-8. For any other token, its bytes are its vocabulary string mapped back through the inverse of the step 3 alphabet. Keep every token, whether or not it is marked `special`. `decode([])` returns `""`.

Both functions must return new objects, support repeated calls in any order, and never mutate `tokenizer_json`. Two tokenizers created by separate calls to `load_tokenizer` must not share mutable state.

Implement the tokenizer yourself. Do not import or use `tokenizers`, `transformers`, or `learnai.kit.bpe.ByteLevelBPE`. Using `translate_pattern` is allowed.

> [!NOTE]
> In Granite's file, the id of the token created by the merge at rank $r$ happens to be $256 + r$. That is a coincidence of this specific vocabulary, not a guarantee of the format. The tests include a file whose ids are shuffled.

## Example

Here is a 10-token tokenizer using Granite's word pattern (omitted for brevity), which matches runs of letters with at most one leading space or other non-letter:

```python
tokenizer_json = {
    "added_tokens": [{"id": 10, "content": "<|eot|>", "special": True, ...}],
    "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "..."}, "behavior": "Removed", "invert": True},
        {"type": "ByteLevel", "add_prefix_space": False, ...}]},
    "model": {
        "type": "BPE",
        "vocab": {"e": 0, "l": 1, "o": 2, "r": 3, "w": 4, "Ġ": 5,
                  "er": 6, "lo": 7, "low": 8, "Ġlow": 9},
        "merges": [["l", "o"], ["lo", "w"], ["e", "r"], ["Ġ", "low"]],
    },
    ...
}
encode, decode = load_tokenizer(tokenizer_json)
encode("low lower<|eot|>")       # [8, 9, 6, 10]
decode([8, 9, 6, 10])            # "low lower<|eot|>"
```

Step 1 finds `<|eot|>` at index 9, leaving `"low lower"` before it and nothing after it. Step 2 splits `"low lower"` into `"low"` and `" lower"`. The leading space stays attached to `"lower"`, and step 3 encodes it as `Ġ`.

| word | list | pair merged |
|---|---|---|
| `low` | `l o w` | `(l, o)`, rank 0 |
| | `lo w` | `(lo, w)`, rank 1 |
| | `low` | none left: id 8 |
| `Ġlower` | `Ġ l o w e r` | `(l, o)`, rank 0 |
| | `Ġ lo w e r` | `(lo, w)`, rank 1 |
| | `Ġ low e r` | `(e, r)`, rank 2 |
| | `Ġ low er` | `(Ġ, low)`, rank 3 |
| | `Ġlow er` | none left: ids 9, 6 |

Appending the added token `<|eot|>` (id 10) gives `[8, 9, 6, 10]`.

With the real Granite 4.0 file:

```python
encode("In 2026")  # [644, 220, 2366, 21]     "In", " ", "202", "6"
encode("<think>")  # [100274]                 added, though not special
```

Granite's pattern groups digits into chunks of at most three and does not attach a leading space to digits, so `" 2026"` splits into three words: `" "`, `"202"`, and `"6"`.

## Constraints

- Token ids and decoded strings are compared exactly against the `tokenizers` library (version 0.23).
- Tests use three files. The first is the 10-token example above. The second is a 456-token miniature of Granite's file with the same pattern, four special tokens, and shuffled ids. The third is the real `tokenizer.json` of `ibm-granite/granite-4.0-350m` (100,352 ids, 100,000 merges, and 96 added tokens, 12 of which are not special). Only the third file requires downloading.
- Input texts have at most 2,000 characters and contain only characters whose bytes are covered by the file's vocabulary. All ids passed to `decode` are valid ids in the file.
- `load_tokenizer` on the real file must finish well under a second after the first call (the first `translate_pattern` call builds Unicode tables and takes about half a second). Encoding 2,000 characters must take a small fraction of a second, so store merge ranks in a dict for $O(1)$ lookups.

## Why this matters

`tokenizer.json` is the contract between training and inference. Every non-Python runtime, from llama.cpp to in-browser engines, has to re-implement the interacting sections of this file, and each section is a common source of silent bugs.

Hardcoding GPT-2's word pattern changes how numbers are split compared to training. Matching only `special` added tokens breaks `<think>` into seven ordinary tokens the model never saw in that role. Ranking merges by id or computing ids from ranks passes on tokenizers that happen to number merges sequentially and breaks on all others. None of these bugs raise an error. The model simply receives different token ids than it was trained on and degrades quietly.
