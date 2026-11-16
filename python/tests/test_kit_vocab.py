"""ToyVocab: the token table, its counters, and its byte-pair encoder."""

import numpy as np
import pytest

from learnai.judge import frameworks
from learnai.kit import TinyLM, ToyVocab
from learnai.kit.vocab import MAX_SIZE, MIN_SIZE, SPECIALS


def test_sizes_and_alignment_with_tinylm():
    assert (MIN_SIZE, MAX_SIZE) == (64, 282)
    vocab = ToyVocab()
    lm = TinyLM(frameworks.get("python"))
    assert vocab.size == len(vocab) == len(vocab.pieces) == lm.vocab_size == 64
    assert (vocab.bos_id, vocab.eos_id, vocab.pad_id) == (lm.bos_id, lm.eos_id, 2)
    for size in (63, 283, 0):
        with pytest.raises(ValueError, match="size"):
            ToyVocab(size=size)


def test_specials_have_fixed_ids_and_no_bytes():
    vocab = ToyVocab(seed=5)
    assert vocab.specials == {
        "<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2, "<|im_start|>": 3, "<|im_end|>": 4,
    }  # fmt: skip
    assert tuple(vocab.specials) == SPECIALS
    for token_id in range(5):
        assert vocab.bytes(token_id) == b"" and vocab.is_special(token_id)
    assert not vocab.is_special(5)
    assert all(vocab.pieces[i] for i in range(5, vocab.size))  # every other token has bytes
    assert vocab.id_of("<|eos|>") == 1


def test_the_table_has_the_awkward_cases():
    vocab = ToyVocab()
    pieces = set(vocab.pieces)
    assert len(pieces) == 60  # 59 distinct ordinary pieces plus b""
    for single in b'0123456789{}[]:," -.aeflnrstu':
        assert bytes([single]) in pieces
    for piece in (b"true", b"false", b"null", b'": ', b", ", b'{"', b'"}'):
        assert piece in pieces
    # Whole multi-byte characters, and pieces that are only part of one.
    assert "é".encode() in pieces and "€".encode() in pieces and "😀".encode() in pieces
    for partial in (b"\xc3", b"\xa9", b"\xe2\x82", b"\xac", b"\xf0\x9f", b"\x98\x80"):
        assert partial in pieces
        with pytest.raises(UnicodeDecodeError):
            partial.decode("utf-8")


def test_seed_shuffles_ordinary_ids_only():
    a, b, c = ToyVocab(seed=0), ToyVocab(seed=0), ToyVocab(seed=1)
    assert a.pieces == b.pieces
    assert a.pieces != c.pieces and sorted(a.pieces) == sorted(c.pieces)
    assert a.pieces[:5] == c.pieces[:5] == (b"",) * 5
    assert a.merges == c.merges


def test_bytes_and_decode_are_counted():
    vocab = ToyVocab()
    true_id, euro = vocab.id_of(b"true"), vocab.id_of("€")
    assert vocab.counters.snapshot() == {"lookups": 0, "bytes_decoded": 0}
    assert vocab.bytes(true_id) == b"true"
    assert vocab.counters.snapshot() == {"lookups": 1, "bytes_decoded": 4}
    assert vocab.decode([true_id, vocab.eos_id, euro]) == "true€"
    assert vocab.counters.snapshot() == {"lookups": 4, "bytes_decoded": 11}
    assert vocab.decode([]) == ""
    vocab.pieces[true_id], vocab.id_of(b"true"), vocab.encode("true"), vocab.is_special(3)
    assert vocab.counters.snapshot() == {"lookups": 4, "bytes_decoded": 11}  # none of those count


def test_redecoding_from_the_start_is_quadratic():
    vocab = ToyVocab()
    ids = vocab.encode('{"a": true, "n": null}' * 4)
    once = sum(len(vocab.pieces[i]) for i in ids)
    for end in range(1, len(ids) + 1):
        vocab.decode(ids[:end])
    assert vocab.counters.bytes_decoded > once * len(ids) / 3
    vocab.counters.reset()
    for token_id in ids:
        vocab.bytes(token_id)
    assert vocab.counters.bytes_decoded == once


def test_multi_byte_characters_split_across_tokens():
    vocab = ToyVocab()
    first, second = vocab.id_of(b"\xe2\x82"), vocab.id_of(b"\xac")
    assert vocab.decode([first]) == "�"  # a cut-off character
    assert vocab.decode([second]) == "�"
    assert vocab.decode([first, second]) == "€"
    halves = [vocab.id_of(b"\xf0\x9f"), vocab.id_of(b"\x98\x80")]
    assert vocab.decode(halves) == "😀"
    # Token-by-token decoding breaks the text; decoding the sequence does not.
    assert "".join(vocab.decode([i]) for i in halves) != "😀"
    four = [vocab.id_of(bytes([b])) for b in "😀".encode()]
    assert vocab.decode(four) == "😀" and vocab.encode("😀") == [vocab.id_of("😀")]


def test_encode_merges_by_rank_and_round_trips():
    vocab = ToyVocab(seed=3)
    name = lambda ids: [vocab.pieces[i] for i in ids]
    assert name(vocab.encode("true")) == [b"true"]
    assert name(vocab.encode("tru")) == [b"tru"]
    assert name(vocab.encode("rut")) == [b"r", b"u", b"t"]
    assert name(vocab.encode('{"a": null}')) == [b'{"', b"a", b'": ', b"null", b"}"]
    # '":' ranks above '"}' and '{"': the quote goes to the earlier merge.
    assert name(vocab.encode('":"}')) == [b'":', b'"}']
    assert name(vocab.encode("é€")) == ["é".encode(), "€".encode()]
    assert vocab.encode("") == [] and vocab.encode(b"10") == vocab.encode("10")
    text = '{"a": [10, -2.5, true, false, null], "é": "€😀"}'
    ids = vocab.encode(text)
    assert vocab.decode(ids) == text
    assert not any(vocab.is_special(i) for i in ids)
    # Special strings in ordinary text are ordinary characters (here: not encodable at all).
    with pytest.raises(ValueError, match="0x3c"):
        vocab.encode("<|eos|>")


def test_merges_are_ranked_pairs_of_existing_pieces():
    vocab = ToyVocab()
    assert len(vocab.merges) == 21 and vocab.merges[0] == (b"t", b"r")
    seen = {piece for piece in vocab.pieces if len(piece) == 1}
    for left, right in vocab.merges:
        assert left in seen and right in seen  # a merge builds on earlier pieces
        seen.add(left + right)
        assert vocab.id_of(left + right) >= 5
    assert seen == set(vocab.pieces) - {b""}


def test_larger_sizes_add_bytes_until_every_byte_has_a_token():
    assert set(ToyVocab(64).pieces) < set(ToyVocab(100).pieces) < set(ToyVocab(282).pieces)
    medium = ToyVocab(size=130, seed=2)
    assert medium.decode(medium.encode("Hello, World! true")) == "Hello, World! true"
    full = ToyVocab(size=MAX_SIZE)
    assert {bytes([b]) for b in range(256)} <= set(full.pieces)
    text = "<|im_start|>naïve 日本語\n\t\x00"
    assert full.decode(full.encode(text)) == text
    assert full.decode(full.encode(bytes(range(256)))) == bytes(range(256)).decode(
        "utf-8", errors="replace"
    )
    with pytest.raises(ValueError, match="no token"):
        ToyVocab(size=281).encode(bytes(range(256)))


def test_bad_ids_and_unknown_pieces():
    vocab = ToyVocab()
    for bad in (-1, 64, "x", None):
        with pytest.raises(ValueError):
            vocab.bytes(bad)
    with pytest.raises(ValueError):
        vocab.decode([3, 64])
    with pytest.raises(ValueError):
        vocab.is_special(99)
    with pytest.raises(KeyError):
        vocab.id_of(b"zzz")
    assert vocab.bytes(np.int64(6)) == vocab.pieces[6]


def test_with_framework_gives_a_twin_with_fresh_counters(fw):
    vocab = ToyVocab(seed=4)
    vocab.bytes(7)
    twin = fw.to_framework(vocab, fresh_kit=True)
    assert twin is not vocab and twin.pieces == vocab.pieces
    assert twin.counters.lookups == 0 and vocab.counters.lookups == 1
    assert fw.to_framework(vocab) is vocab  # a solution call keeps the instance under test
    twin.bytes(8)
    assert vocab.counters.lookups == 1
    if fw.name != "python":
        assert vocab.bytes(fw.tensor(np.array(9))) == vocab.pieces[9]  # a 0-d tensor id
