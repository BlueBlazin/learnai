"""ToyVocab: a small token table with the awkward cases of a real one.

``TinyLM`` produces token ids; a ``ToyVocab`` of the same size says which bytes each
id stands for, so problems about detokenizing, stop strings, and constrained decoding
have text to work with:

    vocab = ToyVocab(size=64, seed=0)
    vocab.bytes(17)                 # b'true', say: the bytes of one token
    vocab.decode([17, 9])           # the text of a token sequence
    vocab.encode('{"a": 10}')       # token ids, by byte-pair merging
    vocab.specials                  # {'<|bos|>': 0, '<|eos|>': 1, ...}

The table is tiny, yet it has what makes real vocabularies awkward:

* **special tokens** (ids 0 to 4: ``<|bos|>``, ``<|eos|>``, ``<|pad|>``,
  ``<|im_start|>``, ``<|im_end|>``). They stand for no text: their bytes are ``b""``;
* **single bytes**: the digits, JSON punctuation, a few letters, and the bytes of the
  UTF-8 encodings of ``é``, ``€``, and ``😀``, which are not characters on their own;
* **multi-byte pieces** built by a ranked merge table (``vocab.merges``): words such as
  ``true`` and ``null``, punctuation runs such as ``": "``, whole characters such as
  ``é``, and pieces that are *part* of a character, such as the first two bytes of ``€``.

So one token can hold several characters, and one character can be spread over several
tokens: decoding token by token breaks text.

Ids 0 to 4 are fixed. The other ids are shuffled by ``seed``, so a solution cannot
hard-code them. Sizes above 64 add more single bytes; from size 282 on every byte value
has a token and ``encode`` accepts any text.

Lookups are counted in ``vocab.counters``: ``lookups`` (token ids looked up) and
``bytes_decoded`` (bytes handed out by ``bytes`` and ``decode``). Decoding a growing
sequence from the start at every step makes ``bytes_decoded`` grow quadratically.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import numpy as np

from whetstone.kit.counters import Counters

#: Special tokens, in id order. The first two match ``TinyLM.bos_id`` and ``eos_id``.
SPECIALS = ("<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>")

#: The single-byte tokens of the smallest vocabulary: digits, JSON punctuation, the
#: letters of true/false/null, and the bytes of "é" (C3 A9), "€" (E2 82 AC), and
#: "😀" (F0 9F 98 80).
_CORE_BYTES = b'0123456789{}[]:," -.aeflnrstu' + bytes.fromhex("c3a9e282acf09f9880")

#: The merge table, best rank first. Each merge joins two existing pieces into a new
#: token; later merges build on earlier ones ("t"+"r", then "tr"+"u", then "tru"+"e").
_MERGES: tuple[tuple[builtins.bytes, builtins.bytes], ...] = (
    (b"t", b"r"), (b"tr", b"u"), (b"tru", b"e"),
    (b"f", b"a"), (b"fa", b"l"), (b"fal", b"s"), (b"fals", b"e"),
    (b"n", b"u"), (b"nu", b"l"), (b"nul", b"l"),
    (b'"', b":"), (b'":', b" "), (b",", b" "), (b"{", b'"'), (b'"', b"}"),
    (b"\xc3", b"\xa9"),                      # é
    (b"\xe2", b"\x82"), (b"\xe2\x82", b"\xac"),          # the start of €, then €
    (b"\xf0", b"\x9f"), (b"\x98", b"\x80"), (b"\xf0\x9f", b"\x98\x80"),  # two halves of 😀
)  # fmt: skip

MIN_SIZE = len(SPECIALS) + len(_CORE_BYTES) + len(_MERGES)  # 64
MAX_SIZE = len(SPECIALS) + 256 + len(_MERGES)  # 282: every byte value has a token


def _extra_bytes() -> list[int]:
    """Single bytes added by sizes above the minimum: printable ASCII first, then the rest."""
    printable = [*range(0x20, 0x7F), 0x0A, 0x09]
    ordered = printable + [b for b in range(256) if b not in printable]
    return [b for b in ordered if b not in _CORE_BYTES]


class ToyVocab:
    """A seeded token table. See the module docstring for what is in it.

    Args:
        size: number of tokens, from 64 to 282. Use the model's ``vocab_size``.
        seed: shuffles the ids of the non-special tokens.

    Attributes:
        size: as given (also ``len(vocab)``).
        specials: special token strings and their ids, ``{"<|bos|>": 0, ...}``.
        bos_id, eos_id, pad_id: 0, 1, 2.
        merges: the merge table as ``(left_bytes, right_bytes)`` pairs, best rank first.
        pieces: the bytes of every token, indexed by id. Reading it is not counted;
            it is meant for tests.
        counters: ``lookups`` and ``bytes_decoded``.
    """

    bos_id = 0
    eos_id = 1
    pad_id = 2

    def __init__(self, size: int = 64, seed: int = 0) -> None:
        if not MIN_SIZE <= size <= MAX_SIZE:
            raise ValueError(f"size must be between {MIN_SIZE} and {MAX_SIZE}, got {size}")
        self.size = size
        self.seed = seed
        self.specials: dict[str, int] = {name: i for i, name in enumerate(SPECIALS)}
        self.merges = _MERGES

        singles = [*_CORE_BYTES, *_extra_bytes()[: size - MIN_SIZE]]
        ordinary = [builtins.bytes([b]) for b in singles] + [left + right for left, right in _MERGES]
        order = np.random.default_rng(seed).permutation(len(ordinary))
        self.pieces: tuple[builtins.bytes, ...] = (
            *(b"" for _ in SPECIALS),
            *(ordinary[i] for i in order),
        )
        self._ids = {piece: i for i, piece in enumerate(self.pieces) if piece}
        self._rank = {
            (self._ids[left], self._ids[right]): (rank, self._ids[left + right])
            for rank, (left, right) in enumerate(_MERGES)
        }
        self.counters = Counters("lookups", "bytes_decoded")

    def with_framework(self, fw: Any) -> ToyVocab:
        """The same table with fresh counters (token ids are plain ints in every framework)."""
        twin = object.__new__(ToyVocab)
        twin.__dict__.update(self.__dict__)
        twin.counters = Counters("lookups", "bytes_decoded")
        return twin

    def __len__(self) -> int:
        return self.size

    # -- ids to bytes and text -----------------------------------------------------

    def bytes(self, token_id: int) -> builtins.bytes:
        """The bytes of one token; ``b""`` for a special token. Counted.

        Raises ``ValueError`` for an id outside ``[0, size)``.
        """
        piece = self.pieces[self._index(token_id)]
        self.counters.add("lookups")
        self.counters.add("bytes_decoded", len(piece))
        return piece

    def decode(self, token_ids: Iterable[int]) -> str:
        """The text of a token sequence: its bytes, joined, read as UTF-8. Counted.

        Special tokens contribute nothing. Bytes that are not valid UTF-8, including a
        character cut off at the end, become U+FFFD, as ``errors="replace"`` does.
        """
        data = b"".join(self.bytes(token_id) for token_id in token_ids)
        return data.decode("utf-8", errors="replace")

    def is_special(self, token_id: int) -> bool:
        """Whether ``token_id`` is a special token."""
        return self._index(token_id) < len(SPECIALS)

    # -- bytes and text to ids ------------------------------------------------------

    def id_of(self, piece: builtins.bytes | str) -> int:
        """The id of the token whose bytes are exactly ``piece``, or of a special token
        given by its string. Raises ``KeyError`` if there is none."""
        if isinstance(piece, str):
            if piece in self.specials:
                return self.specials[piece]
            piece = piece.encode("utf-8")
        return self._ids[piece]

    def encode(self, text: str | builtins.bytes) -> list[int]:
        """Ordinary text to token ids by byte-pair merging; never yields a special token.

        Starts from one token per byte, then repeatedly takes, among the adjacent pairs
        present, the one that comes first in ``merges`` and replaces every occurrence
        of it, left to right. Special token strings in ``text`` are encoded as ordinary
        characters. Raises ``ValueError`` if a byte has no token (possible below size
        282). Not counted.
        """
        data = text.encode("utf-8") if isinstance(text, str) else builtins.bytes(text)
        ids = []
        for byte in data:
            single = builtins.bytes([byte])
            if single not in self._ids:
                raise ValueError(f"byte 0x{byte:02x} has no token in this vocabulary")
            ids.append(self._ids[single])
        while len(ids) >= 2:
            best = min(pairwise(ids), key=lambda pair: self._rank.get(pair, (len(_MERGES),)))
            if best not in self._rank:
                break
            merged, out, i = self._rank[best][1], [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    # -- helpers -------------------------------------------------------------------

    def _index(self, token_id: Any) -> int:
        try:
            index = int(token_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token id must be an integer, got {token_id!r}") from exc
        if not 0 <= index < self.size:
            raise ValueError(f"token id {index} is outside [0, {self.size})")
        return index

    def __repr__(self) -> str:
        return f"ToyVocab(size={self.size}, seed={self.seed})"
