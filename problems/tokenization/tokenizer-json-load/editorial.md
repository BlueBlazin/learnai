## The idea

A `tokenizer.json` file describes a staged pipeline whose sections run in order: `added_tokens`, `normalizer`, `pre_tokenizer`, `model`, `post_processor`, and, in reverse, `decoder`. Because Granite's file leaves the normalizer and post-processor empty, encoding is a four-stage pipeline:

$$
\text{text} \xrightarrow{\text{added tokens}} \text{stretches}
\xrightarrow{\text{Split regex}} \text{words}
\xrightarrow{\text{ByteLevel}} \text{characters}
\xrightarrow{\text{BPE}} \text{ids}.
$$

Each stage protects the next stage from breaking token boundaries:

- **Added tokens run first** on the raw text so downstream stages never see them. The word regex would split `<|end_of_text|>` into `<|`, `end`, `_of`, `_text`, and `|>`, and no sequence of BPE merges can reassemble it. Its entry in `vocab` is unreachable by BPE on purpose. The `special` flag does not change matching. It only tells a decoder whether to hide the token when skipping special tokens. Tokens like `<think>` and `<tool_call>` are added and atomic even though `special` is `false`.
- **Words bound the merges.** Running BPE inside each word keeps merges cheap and cacheable, and ensures a word always gets the same ids regardless of context. The split pattern is model data, not a constant. Granite 4.0 uses the tiktoken `cl100k` pattern (a word takes one leading non-letter, digits split into groups of three, and spaces do not attach to digits), whereas Granite 4.2 uses GPT-2's pattern. With the same 100,000 merges, `"In 2026"` encodes to `[644, 220, 2366, 21]` under Granite 4.0's pattern and `[644, 220, 508, 1627]` under GPT-2's.
- **ByteLevel makes the vocabulary printable.** BPE operates on raw bytes, but JSON object keys must be valid Unicode strings. ByteLevel maps each byte to itself if it is printable in Latin-1, and to a character from U+0100 upward otherwise. That is why a leading space appears as `Ġ` and `é` (bytes `C3 A9`) appears as `Ã©` in `vocab`.
- **BPE** runs the same merge loop as `bpe-encode`, but over strings instead of integer ids. Each merged token is the concatenation of its two parts, and its id is looked up in `vocab`.

Decoding inverts only the last two stages, mapping ids to bytes and decoding those bytes as UTF-8. Word and added-token boundaries leave no trace in the id sequence and are not needed to reconstruct the text.

## Reference solution (Python)

```python
def load_tokenizer(tokenizer_json):
    model = tokenizer_json["model"]
    vocab = model["vocab"]
    rank = {(left, right): i for i, (left, right) in enumerate(model["merges"])}
    added = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
    by_length = sorted(added, key=len, reverse=True)
    added_re = re.compile("|".join(map(re.escape, by_length))) if added else None
    pattern = tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    word_re = re.compile(translate_pattern(pattern))

    alphabet = byte_alphabet()  # 256 characters, see the hints
    byte_of = {char: byte for byte, char in enumerate(alphabet)}
    token_bytes = {i: bytes(byte_of[c] for c in token) for token, i in vocab.items()}
    token_bytes.update({i: content.encode("utf-8") for content, i in added.items()})

    def bpe(word):
        parts = list(word)
        while len(parts) > 1:
            best = min(pairwise(parts), key=lambda pair: rank.get(pair, len(rank)))
            if best not in rank:
                break
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return [vocab[part] for part in parts]

    def encode_ordinary(text):
        ids = []
        for match in word_re.finditer(text):
            word = "".join(alphabet[byte] for byte in match.group().encode("utf-8"))
            ids.extend(bpe(word))
        return ids

    def encode(text):
        if added_re is None:
            return encode_ordinary(text)
        ids, position = [], 0
        for match in added_re.finditer(text):
            ids.extend(encode_ordinary(text[position : match.start()]))
            ids.append(added[match.group()])
            position = match.end()
        ids.extend(encode_ordinary(text[position:]))
        return ids

    def decode(ids):
        return b"".join(token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    return encode, decode
```

Added tokens that also appear in `vocab` are written into `token_bytes` twice: first via the byte alphabet, then overwritten with `content.encode("utf-8")`. The second assignment always wins, and for ASCII tokens both produce the same bytes.

## Common mistakes

- **Ranking merges by vocabulary id.** In tiktoken-derived vocabularies like Granite's, token $256 + r$ happens to be the result of merge $r$, so "lowest id first" and "lowest rank first" agree and hide the bug. The format makes no such guarantee. Always use the pair's index in `merges` as its rank.
- **Computing ids as `256 + rank`.** This is the same assumption in reverse, carried over from tokenizers defined only by a merge list. Always look up ids in `vocab`.
- **Matching added tokens after pre-tokenization.** By the time the word regex runs, added tokens have already been split into fragments. Match added tokens on the raw text first.
- **Matching only `special` added tokens.** Twelve of Granite's 96 added tokens have `"special": false`, including `<think>`, `<tool_call>`, and `<|fim_prefix|>`. They are still atomic tokens. A miniature test file whose added tokens are all special will not catch this bug, but the real file will.
- **Picking the shorter added token at the same position.** The rule is leftmost, then longest. In a Python regex alternation, the first matching branch wins, so you must sort escaped alternatives from longest to shortest.
- **Hardcoding the GPT-2 pattern.** Many byte-level BPE files set `"pre_tokenizer": {"type": "ByteLevel", "use_regex": true}` to use GPT-2's pattern. Granite supplies its own `Split` regex and disables ByteLevel's built-in regex, so you must read the pattern from the file.
- **Using an incomplete byte alphabet.** Mapping only space to `Ġ`, newline to `Ċ`, and everything else to `chr(byte)` works on ASCII but fails on non-Latin text. Bytes `0x80`–`0xA0` appear in most multi-byte UTF-8 characters (for example, `東` is `E6 9D B1`) and map to code points from U+0100 upward.
- **Merging the leftmost mergeable pair.** Always merge the pair with the lowest rank in `merges`, not the first mergeable pair from the left.
- **Decoding token by token.** A multi-byte character like `🙂` (4 bytes) can span two or three tokens. Decoding each token separately produces U+FFFD replacement characters. Concatenate all token bytes first, then decode once.
- **Dropping special tokens in `decode`.** The `tokenizers` library defaults to `skip_special_tokens=True` in `Tokenizer.decode`, which is a display convenience rather than the raw decoding of the id sequence. Keep all tokens here.

## Complexity

Loading takes time linear in the size of `tokenizer_json`. With this BPE loop, encoding a word of $m$ bytes takes $O(m^2)$ time in the worst case ($m$ merge rounds, each scanning the token list). Because words are short, total encoding time is nearly linear in the input length. Production tokenizers cache the encoded ids of each distinct word for amortized $O(1)$ lookup per repeated word, and use a priority queue for long words.

## In the wild

The `tokenizers` library implements this exact pipeline in Rust. It uses an Aho-Corasick automaton for leftmost-longest added-token matching and the Oniguruma regex engine for pre-tokenization. Oniguruma supports `\p{L}` and defines `\s` as Unicode `White_Space` (slightly narrower than Python's `\s`), which `translate_pattern` bridges for Python's `re`.

By contrast, `tiktoken` stores a vocabulary as ranked byte strings without a separate merge list. It merges the adjacent pair whose *concatenation* has the lowest rank, which is valid there because the format defines ids by merge order. Converting a tiktoken vocabulary into `tokenizer.json` creates the exact $256 + r$ coincidence noted in the statement. Upstream, chat templates (`chat-template-render`) emit the strings that step 1 turns into role tokens, while `special-tokens-split` handles the policy of when user input is allowed to contain them.
