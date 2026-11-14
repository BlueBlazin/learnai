## The idea

A `tokenizer.json` describes a pipeline, and the file's sections are its stages in
order: `added_tokens`, `normalizer`, `pre_tokenizer`, `model`, `post_processor`, and,
for the way back, `decoder`. Granite's file leaves the normalizer and the
post-processor empty, so encoding is four nested loops:

$$
\text{text} \xrightarrow{\text{added tokens}} \text{stretches}
\xrightarrow{\text{Split regex}} \text{words}
\xrightarrow{\text{ByteLevel}} \text{characters}
\xrightarrow{\text{BPE}} \text{ids}.
$$

Each stage exists to stop the next one from doing something wrong.

- **Added tokens first**, on the raw text, because nothing downstream may see them.
  The pattern would cut `<|end_of_text|>` into `<|`, `end`, `_of`, `_text`, `|>`, and
  BPE would never put it back together: there is no chain of merges that builds it.
  Its entry in `vocab` is unreachable by design. The `special` flag does not change
  this. It only says whether a decoder asked to skip special tokens drops the token;
  `<think>` and `<tool_call>` are added, atomic, and not special.
- **Words** bound the merges. BPE inside a word is cheap and cacheable, and the same
  word always gets the same ids wherever it appears. The pattern is the tokenizer's
  opinion about what a word is, and it is data: Granite 4.0 uses the tiktoken
  `cl100k` pattern (a word takes one leading non-letter with it, digits come in
  groups of three, a space is not attached to digits), while Granite 4.2 uses GPT-2's.
  The same 100,000 merges then give `[644, 220, 2366, 21]` for `"In 2026"` under one
  and `[644, 220, 508, 1627]` under the other.
- **ByteLevel** makes the vocabulary printable. BPE works on bytes, but JSON keys are
  strings, so each byte gets a character: itself if it is printable in Latin-1, and a
  character from U+0100 on otherwise. That is why a leading space shows up as `Ġ` and
  why `é`, the bytes `C3 A9`, is spelled `Ã©` in the vocabulary.
- **BPE** is the loop from `bpe-encode` over strings instead of ints: the merged
  symbol is the concatenation, and its id is whatever `vocab` says.

Decoding is the inverse of the last two stages only: ids to bytes, bytes to text. The
word boundaries and the added-token boundaries leave no trace in the ids, and need
none.

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

The added tokens that the file also lists in `vocab` get their bytes twice, first
through the alphabet and then from `content`; the second assignment wins and, for
ASCII contents, is the same.

## Common mistakes

- **Ranking merges by vocabulary id.** In a tiktoken-derived vocabulary such as
  Granite's, token $256 + r$ is the product of merge $r$, so "lowest id first" and
  "lowest rank first" agree and the bug is invisible. The format promises no such
  thing: the rank is the position in `merges`.
- **Computing ids as `256 + rank`.** The same coincidence from the other side, and a
  habit from tokenizers that are defined by their merge list alone. Look ids up.
- **Looking for added tokens after pre-tokenization**, or only as whole words. By
  then they are in pieces. They are matched on the raw text, before anything else.
- **Matching only `special` added tokens.** In Granite's file 12 of the 96 added
  tokens are not special, among them `<think>`, `<tool_call>`, and `<|fim_prefix|>`.
  They are still single tokens. A miniature whose added tokens are all special cannot
  show this; the real file does.
- **Taking the shorter of two added tokens at one position.** Leftmost, then longest;
  an alternation regex gets this right only if the alternatives are sorted by length,
  longest first.
- **Assuming the GPT-2 pattern.** Most byte-level BPE files say
  `"pre_tokenizer": {"type": "ByteLevel", "use_regex": true}`, which means GPT-2's
  pattern. This one carries its own `Split` regex and switches ByteLevel's off. Read
  the file.
- **An incomplete byte alphabet.** Mapping the space to `Ġ`, the newline to `Ċ`, and
  everything else to `chr(byte)` works for ASCII text. The bytes `0x80` to `0xA0`
  occur in almost every non-Latin character (`東` is `E6 9D B1`), and they have their
  own characters.
- **Merging the leftmost mergeable pair** instead of the pair of lowest rank: the bug
  `bpe-encode` was about, still available here.
- **Decoding token by token.** `🙂` is four bytes that may sit in two or three tokens;
  decoding each alone yields replacement characters. Join the bytes, decode once.
- **Dropping special tokens in `decode`.** `Tokenizer.decode` in the `tokenizers`
  library does that by default (`skip_special_tokens=True`). It is a display option,
  not part of what the ids mean.

## Complexity

Loading is linear in the size of the file. Encoding a word of $m$ bytes costs
$O(m^2)$ in the worst case with this loop ($m$ rounds, each a pass over the list);
words are short, so the total is close to linear in the text. Production tokenizers
cache the ids of each distinct word, which makes the amortized cost a dictionary
lookup per word, and use a priority queue inside long words.

## In the wild

The `tokenizers` library runs exactly this pipeline in Rust, with an Aho-Corasick
automaton for the added tokens (leftmost-longest, as specified here) and the Oniguruma
regex engine for the pattern, which is why the pattern speaks `\p{L}` and why `\s`
means Unicode White_Space rather than Python's slightly wider class;
`translate_pattern` papers over both. tiktoken stores the same vocabulary as a list
of byte strings with ranks and no merge list at all: it merges the adjacent pair
whose *concatenation* has the lowest rank, which is the "rank by id" rule, valid there
because the format defines ids that way. Converting between the two formats is
exactly the coincidence the note in the statement warns about. Chat templates
(`chat-template-render`) produce the strings that step 1 then turns into role tokens,
and `special-tokens-split` is the policy layer deciding when user text may contain
them.
