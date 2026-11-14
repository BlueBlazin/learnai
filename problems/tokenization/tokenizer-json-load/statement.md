Build a working tokenizer from the `tokenizer.json` of a real model, IBM's Granite 4.0
350M, and reproduce the ids of the `tokenizers` library exactly. The BPE loop is the
one you wrote in `bpe-encode`. What is new is everything a real file wraps around it:
added tokens, a pre-tokenizer that decides where merges may not cross, a printable
alphabet for bytes, and ids that are looked up instead of computed.

## Specification

```python
def load_tokenizer(
    tokenizer_json: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]: ...
```

`tokenizer_json` is the parsed file. `load_tokenizer` returns two functions,
`(encode, decode)`. The parts of the file that matter:

| path in the JSON | content |
|---|---|
| `["model"]["vocab"]` | dict from token string to id |
| `["model"]["merges"]` | list of pairs `[left, right]` of token strings; the position in the list is the pair's **rank** |
| `["added_tokens"]` | list of objects with `"content"` (the token's text), `"id"`, and `"special"` (a bool) |
| `["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]` | the regular expression that cuts text into words |

The rest of the file is the same in every test and needs no handling: there is no
normalizer and no post-processor, the second pre-tokenizer is `ByteLevel` without a
prefix space, and every added token has `lstrip`, `rstrip`, `single_word`, and
`normalized` set to `false`.

**`encode(text)`** returns the list of ids produced by these four steps.

1. **Added tokens.** Scan `text` from left to right for the `content` of any added
   token, **whether or not it is flagged `special`**. The leftmost occurrence wins;
   among added tokens that start at the same position, the longest wins. Each
   occurrence becomes the token's `id`, and the scan continues after it. The stretches
   of text between occurrences (and before the first, and after the last) are encoded
   by steps 2 to 4, each on its own; empty stretches produce nothing.
2. **Words.** Cut the stretch into words with the pattern: the words are its
   successive non-overlapping matches from left to right, as `re.finditer` yields
   them. Merges never cross a word boundary. The pattern uses `\p{L}` and `\p{N}`,
   which Python's `re` does not know, so compile it with the kit:
   `re.compile(translate_pattern(pattern))`, where `translate_pattern` comes from
   `whetstone.kit.bpe`. Read the pattern from the file.
3. **Bytes as characters.** Encode the word as UTF-8 and write every byte as one
   character of the ByteLevel alphabet, which is how the vocabulary spells tokens:
   - the 188 bytes `0x21` to `0x7E`, `0xA1` to `0xAC`, and `0xAE` to `0xFF` are written
     as the character with the same code point (`chr(byte)`);
   - the other 68 bytes, taken in increasing order, are written as U+0100, U+0101,
     and so on. So byte `0x00` is `Ā` (U+0100), the newline `0x0A` is `Ċ` (U+010A),
     the space `0x20` is `Ġ` (U+0120), and `0x7F` is `ġ` (U+0121).
4. **BPE.** Start with the word as a list of single characters. Repeat: among the
   adjacent pairs of the current list that appear in `merges`, take the one with the
   lowest rank; replace every occurrence of it, left to right and without overlap, by
   the concatenation of its two strings. Stop when no adjacent pair is in `merges`.
   The ids of the word are `vocab[s]` for each string `s` that remains.

Nothing is added around the text (no beginning-of-text token). `encode("")` is `[]`.

**`decode(ids)`** returns the text of a list of ids: the bytes of every token, in
order, joined and then decoded as UTF-8 with invalid sequences replaced by U+FFFD
(`errors="replace"`). The bytes of an added token are its `content` in UTF-8; the
bytes of any other token are its vocabulary string read back through the alphabet of
step 3. Every token is kept, special or not. `decode([])` is `""`.

Both functions return new objects, may be called any number of times in any order,
and must not modify `tokenizer_json`. Two tokenizers loaded from two files must not
share state.

Do the work yourself: the `tokenizers` and `transformers` packages and the kit's
`ByteLevelBPE` are not allowed. `translate_pattern` is.

> [!NOTE]
> In Granite's file the id of the token made by the merge of rank $r$ happens to be
> $256 + r$. That is a property of this vocabulary, not of the format, and the tests
> include a file where ids are in no particular order.

## Example

A complete tokenizer with ten tokens (the pattern, not shown, is Granite's: a word is
a run of letters with at most one leading space or other non-letter):

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

Step 1 finds `<|eot|>` at index 9, which leaves the stretch `"low lower"` before it
and nothing after it. Step 2 cuts the stretch into the words `"low"` and `" lower"`.
The space belongs to the word that follows it, and step 3 writes it as `Ġ`.

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

Then comes the added token, id 10. With the real file:

```python
encode("In 2026")  # [644, 220, 2366, 21]     "In", " ", "202", "6"
encode("<think>")  # [100274]                 added, though not special
```

Granite's pattern cuts digit runs into groups of at most three and does not attach a
space to digits, so `" 2026"` is three words: `" "`, `"202"`, `"6"`.

## Constraints

- Ids and decoded strings are compared exactly with the output of the `tokenizers`
  library (version 0.23).
- Three files are used: the example above; a 456-token miniature of Granite's file
  with the same pattern, four special tokens, and shuffled ids; and the real
  `tokenizer.json` of `ibm-granite/granite-4.0-350m` (100,352 ids, 100,000 merges, 96
  added tokens, 12 of them not special). Only the last needs the download.
- Texts are at most 2,000 characters and contain only characters whose bytes the
  file's vocabulary covers. Ids passed to `decode` are ids of the file.
- `load_tokenizer` on the real file must take well under a second after the first
  call (the first `translate_pattern` call builds Unicode tables and takes about half
  a second), and encoding 2,000 characters a small fraction of a second: look ranks up
  in a dict.

## Why this matters

`tokenizer.json` is the contract between training and inference: every runtime that
is not Python, from llama.cpp to an in-browser engine, has to re-implement what this
file means, and the meaning is spread over five sections that interact. Each one has
a classic port bug. Assume GPT-2's word pattern and numbers are split differently
from training. Treat only `special` tokens as atomic and `<think>` reaches the model
as seven ordinary tokens it has never seen in that role. Rank merges by id, or compute
ids from ranks, and the port works for exactly the vocabularies you tested. None of
these raises an error. The model just reads slightly different ids than it was
trained on, and gets quietly worse.
