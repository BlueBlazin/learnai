# BUG: writes every byte as the character with the same code point, except the space (Ġ) and the newline (Ċ); the other control bytes and the bytes 0x7F to 0xA0 and 0xAD have their own characters in the ByteLevel alphabet.
import re
from collections.abc import Callable
from itertools import pairwise

from whetstone.kit.bpe import translate_pattern


def byte_alphabet() -> list[str]:
    """The character ByteLevel writes each of the 256 byte values as."""
    chars = [chr(byte) for byte in range(256)]
    chars[0x20] = "Ġ"
    chars[0x0A] = "Ċ"
    return chars


def load_tokenizer(
    tokenizer_json: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    model = tokenizer_json["model"]
    vocab: dict[str, int] = model["vocab"]
    # The position of a pair in the merge list is its rank: lower merges first.
    rank = {(left, right): i for i, (left, right) in enumerate(model["merges"])}
    # Every added token is cut out of the text, special or not.
    added = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}
    # Longest first: at one position, regex alternation takes the first branch that fits.
    by_length = sorted(added, key=len, reverse=True)
    added_re = re.compile("|".join(map(re.escape, by_length))) if added else None
    pattern = tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    word_re = re.compile(translate_pattern(pattern))  # read from the file, not assumed

    alphabet = byte_alphabet()
    byte_of = {char: byte for byte, char in enumerate(alphabet)}
    token_bytes = {
        i: bytes(byte_of.get(c, ord(c) & 0xFF) for c in token) for token, i in vocab.items()
    }
    token_bytes.update({i: content.encode("utf-8") for content, i in added.items()})

    def bpe(word: str) -> list[int]:
        """Ids of one word, given as ByteLevel characters."""
        parts = list(word)
        while len(parts) > 1:
            # Of the adjacent pairs present now, the one earliest in the merge list.
            best = min(pairwise(parts), key=lambda pair: rank.get(pair, len(rank)))
            if best not in rank:
                break
            merged, i = [], 0
            while i < len(parts):  # every occurrence, left to right, without overlap
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return [vocab[part] for part in parts]  # the id is looked up, never computed

    def encode_ordinary(text: str) -> list[int]:
        ids: list[int] = []
        for match in word_re.finditer(text):  # the words, left to right
            word = "".join(alphabet[byte] for byte in match.group().encode("utf-8"))
            ids.extend(bpe(word))
        return ids

    def encode(text: str) -> list[int]:
        if added_re is None:
            return encode_ordinary(text)
        ids: list[int] = []
        position = 0
        for match in added_re.finditer(text):  # leftmost, then longest
            ids.extend(encode_ordinary(text[position : match.start()]))
            ids.append(added[match.group()])
            position = match.end()
        ids.extend(encode_ordinary(text[position:]))
        return ids

    def decode(ids: list[int]) -> str:
        # Join the bytes first: one character may be spread over several tokens.
        return b"".join(token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    return encode, decode
