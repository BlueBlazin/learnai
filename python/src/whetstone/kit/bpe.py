"""A byte-level BPE tokenizer read from a Hugging Face ``tokenizer.json``, in plain Python.

    tok = ByteLevelBPE.from_file(path_to_tokenizer_json)
    ids = tok.encode("Hello, world")        # list[int]; no special tokens are added
    tok.decode(ids)                         # "Hello, world"
    tok.token_bytes(ids[0])                 # b"Hello"
    tok.specials["<|end_of_text|>"]         # id of a named special token

This is the kit's reference for the real models' tokenizer: it needs no extra package
at run time, and its ids equal those of the ``tokenizers`` library (the judge's own
tests compare the two when that library is installed).

What it implements is exactly what the Granite and Qwen ``tokenizer.json`` files use,
in the order the ``tokenizers`` pipeline applies it:

1. **Added tokens** (``<|end_of_text|>``, ``<think>``, ...) are cut out of the text
   first, leftmost and longest first; each becomes its own id and is never merged.
2. An optional **NFC normalizer** applies to the text between them.
3. A **pre-tokenizer** cuts the remaining text into words with one regular
   expression. Either a ``Split`` step carrying its own pattern (Granite 4.0 and Qwen
   use the tiktoken ``cl100k`` pattern: contractions, letter runs with one leading
   non-letter, digit runs of at most three, punctuation runs, whitespace), or
   ``ByteLevel`` with ``use_regex`` (Granite 4.2), which means the GPT-2 pattern
   ``GPT2_PATTERN``. The two cut numbers and leading spaces differently, so the same
   vocabulary gives different ids: read the file, do not assume.
4. **ByteLevel**: each word's UTF-8 bytes are written as 256 printable characters (a
   space becomes ``Ġ``), so the vocabulary is plain strings.
5. **BPE**: within a word, the adjacent pair with the lowest merge rank is merged,
   repeatedly, until no pair has a rank.

A ``tokenizer.json`` that uses anything else (another normalizer, a post-processor
that inserts tokens, added tokens with ``lstrip``) is refused with a message saying
what.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import cache
from pathlib import Path
from typing import Any

#: Unicode White_Space, which is what ``\s`` means in the regex engine ``tokenizers``
#: uses. Python's ``\s`` also matches the separators U+001C to U+001F, so the pattern
#: is rewritten with this explicit class.
_WHITE_SPACE = r"\t\n\x0b\x0c\r \x85\xa0  -     　"


#: What ``ByteLevel`` with ``use_regex: true`` splits on (the GPT-2 pattern).
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


@cache
def _category_class(letter: str) -> str:
    r"""The body of a character class for ``\p{L}`` or ``\p{N}``: every code point
    whose Unicode general category starts with ``letter``, as ranges."""
    ranges: list[str] = []
    start = None
    for code in range(sys.maxunicode + 2):
        inside = code <= sys.maxunicode and unicodedata.category(chr(code)).startswith(letter)
        if inside and start is None:
            start = code
        elif not inside and start is not None:
            first, last = f"\\U{start:08x}", f"\\U{code - 1:08x}"
            ranges.append(first if start == code - 1 else f"{first}-{last}")
            start = None
    return "".join(ranges)


def translate_pattern(pattern: str) -> str:
    r"""Rewrite a ``tokenizers`` Split regex for Python's ``re``.

    Python has no ``\p{L}`` and ``\p{N}``, and its ``\s`` is slightly wider than
    Unicode White_Space; all three become explicit character classes. Quantifiers such
    as ``\p{N}{1,3}`` and everything else pass through unchanged.
    """
    out: list[str] = []
    in_class = False
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if nxt == "p" and pattern[i + 2 : i + 3] == "{":
                end = pattern.index("}", i)
                name = pattern[i + 3 : end]
                if name not in ("L", "N"):
                    raise ValueError(
                        f"unsupported Unicode class \\p{{{name}}} in the Split pattern"
                    )
                body = _category_class(name)
                out.append(body if in_class else f"[{body}]")
                i = end + 1
                continue
            if nxt == "s":
                out.append(_WHITE_SPACE if in_class else f"[{_WHITE_SPACE}]")
            elif nxt == "S":
                if in_class:
                    raise ValueError("\\S inside a character class is not supported")
                out.append(f"[^{_WHITE_SPACE}]")
            else:
                out.append(ch + nxt)
            i += 2
            continue
        if ch == "[" and not in_class:
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        out.append(ch)
        i += 1
    return "".join(out)


def byte_alphabet() -> list[str]:
    """The 256 characters ByteLevel writes bytes as (the GPT-2 table).

    Printable Latin-1 characters stand for themselves; the other bytes (controls, the
    space, 0x7F to 0xA0, 0xAD) are mapped to U+0100 onward, so a space is ``Ġ``.
    """
    keep = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    table = {byte: chr(byte) for byte in keep}
    extra = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + extra)
            extra += 1
    return [table[byte] for byte in range(256)]


#: The post-processor template that adds nothing: just the sequence itself.
_SEQUENCE_A = {"Sequence": {"id": "A", "type_id": 0}}


class ByteLevelBPE:
    """See the module docstring. Build with ``from_file`` or ``from_json``."""

    def __init__(self, spec: dict[str, Any]) -> None:
        model = spec.get("model") or {}
        if model.get("type") != "BPE":
            raise ValueError(f"unsupported tokenizer model {model.get('type')!r}; expected BPE")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            raise ValueError("BPE with word prefixes or suffixes is not supported")
        if model.get("dropout") or model.get("byte_fallback") or model.get("ignore_merges"):
            raise ValueError("BPE dropout, byte_fallback, and ignore_merges are not supported")
        normalizer = spec.get("normalizer")
        if normalizer is not None and normalizer.get("type") != "NFC":
            raise ValueError(f"unsupported normalizer {normalizer.get('type')!r}")
        self._nfc = normalizer is not None
        post = spec.get("post_processor") or {}
        if post.get("type") == "TemplateProcessing" and (
            post.get("special_tokens") or post.get("single") != [_SEQUENCE_A]
        ):
            raise ValueError("a post_processor that inserts special tokens is not supported")
        if post.get("type") not in (None, "ByteLevel", "TemplateProcessing"):
            raise ValueError(f"unsupported post_processor {post.get('type')!r}")
        self._split = self._read_pre_tokenizer(spec.get("pre_tokenizer"))

        #: Token string (bytes written in the ByteLevel alphabet) -> id.
        self.vocab: dict[str, int] = dict(model["vocab"])
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        #: (left, right) -> rank; the lowest rank merges first.
        self.ranks: dict[tuple[str, str], int] = {pair: rank for rank, pair in enumerate(merges)}
        alphabet = byte_alphabet()
        self._to_char = alphabet
        self._to_byte = {char: byte for byte, char in enumerate(alphabet)}

        #: Added token content -> id, for every added token.
        self.added: dict[str, int] = {}
        #: The subset flagged ``special`` (end of text, role markers, padding).
        self.specials: dict[str, int] = {}
        for token in spec.get("added_tokens") or []:
            if token.get("lstrip") or token.get("rstrip") or token.get("single_word"):
                raise ValueError(f"added token {token['content']!r} uses lstrip/rstrip/single_word")
            if token.get("normalized") and self._nfc:
                raise ValueError(f"added token {token['content']!r} is normalized")
            self.added[token["content"]] = token["id"]
            if token.get("special"):
                self.specials[token["content"]] = token["id"]
        self._added_by_id = {i: content for content, i in self.added.items()}
        self._by_id = {i: text for text, i in self.vocab.items()}
        # Leftmost match; among matches at one position the longest (alternation order).
        ordered = sorted(self.added, key=len, reverse=True)
        self._added_re = re.compile("|".join(map(re.escape, ordered))) if ordered else None
        self._cache: dict[str, list[int]] = {}

    @staticmethod
    def _read_pre_tokenizer(pre: dict[str, Any] | None) -> re.Pattern[str]:
        pre = pre or {}
        if pre.get("type") == "ByteLevel":
            if pre.get("add_prefix_space") or not pre.get("use_regex"):
                raise ValueError("ByteLevel alone is supported only with use_regex and no prefix")
            return re.compile(translate_pattern(GPT2_PATTERN))
        steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
        if not steps or [s.get("type") for s in steps] != ["Split", "ByteLevel"]:
            raise ValueError(
                "unsupported pre_tokenizer: expected ByteLevel or Sequence[Split, ByteLevel]"
            )
        split, byte_level = steps
        # Both spellings found in the wild keep exactly the regex matches as words:
        # "Removed" with invert (Granite) and "Isolated" without (Qwen; the pattern
        # covers every character, so there is nothing between matches).
        mode = (split.get("behavior"), bool(split.get("invert")))
        if mode not in (("Removed", True), ("Isolated", False)) or "Regex" not in split["pattern"]:
            raise ValueError(f"unsupported Split pre-tokenizer {mode}")
        if byte_level.get("add_prefix_space") or byte_level.get("use_regex"):
            raise ValueError("ByteLevel with add_prefix_space or use_regex is not supported")
        return re.compile(translate_pattern(split["pattern"]["Regex"]))

    @classmethod
    def from_json(cls, spec: dict[str, Any]) -> ByteLevelBPE:
        return cls(spec)

    @classmethod
    def from_file(cls, path: str | Path) -> ByteLevelBPE:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def vocab_size(self) -> int:
        """Number of ids, added tokens included."""
        return max(max(self.vocab.values()), max(self.added.values(), default=-1)) + 1

    # -- encoding ------------------------------------------------------------------

    def _bpe(self, word: str) -> list[int]:
        """Ids of one pre-tokenized word (already in the ByteLevel alphabet)."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        parts = list(word)
        ranks = self.ranks
        while len(parts) > 1:
            best_rank, best = None, -1
            for i in range(len(parts) - 1):
                rank = ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best = rank, i
            if best_rank is None:
                break
            left, right = parts[best], parts[best + 1]
            merged: list[str] = []
            i = 0
            while i < len(parts):  # merge every occurrence of the pair, left to right
                if i < len(parts) - 1 and parts[i] == left and parts[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = [self.vocab[part] for part in parts]
        if len(self._cache) < 100_000:
            self._cache[word] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        if self._nfc:
            text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        to_char = self._to_char
        for word in self._split.findall(text):
            ids.extend(self._bpe("".join(to_char[byte] for byte in word.encode("utf-8"))))
        return ids

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text``. Added tokens in the text become their ids; nothing
        (no beginning- or end-of-text token) is added around it."""
        if self._added_re is None:
            return self._encode_plain(text)
        ids: list[int] = []
        position = 0
        for match in self._added_re.finditer(text):
            if match.start() > position:
                ids.extend(self._encode_plain(text[position : match.start()]))
            ids.append(self.added[match.group()])
            position = match.end()
        if position < len(text):
            ids.extend(self._encode_plain(text[position:]))
        return ids

    # -- decoding ------------------------------------------------------------------

    def token_bytes(self, token_id: int) -> bytes:
        """The bytes one id stands for (an added token: its text, UTF-8 encoded).

        A token may hold part of a multi-byte character; decode a stream by joining
        bytes first (``decode``), not by decoding ids one at a time.
        """
        token_id = int(token_id)
        if token_id in self._added_by_id:
            return self._added_by_id[token_id].encode("utf-8")
        try:
            text = self._by_id[token_id]
        except KeyError:
            raise KeyError(f"token id {token_id} is not in the vocabulary") from None
        return bytes(self._to_byte[char] for char in text)

    def decode(self, ids: Any, *, skip_special_tokens: bool = False) -> str:
        """Text of a sequence of ids. Bytes that do not form valid UTF-8 become U+FFFD."""
        special = set(self.specials.values()) if skip_special_tokens else ()
        data = b"".join(self.token_bytes(i) for i in ids if int(i) not in special)
        return data.decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"ByteLevelBPE(vocab_size={self.vocab_size}, added={len(self.added)})"
