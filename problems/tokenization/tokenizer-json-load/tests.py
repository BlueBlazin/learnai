"""Tests for tokenizer-json-load.

Every expected id and string comes from the `tokenizers` library (`fixtures/make.py`),
never from the reference. Three tokenizer files are used:

- the ten-token example of the statement (`fixtures/example.json`);
- a miniature of Granite's file with 456 tokens, whose ids are shuffled so that an id
  is not 256 plus the merge rank (`fixtures/mini-tokenizer.json`, `mini-cases.json`);
- the real `tokenizer.json` of granite-4.0-350m, which is the only one that needs the
  download (`fixtures/granite-4.0-350m.json` holds its cases).
"""

import copy
import json

from learnai.judge import case, check

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"

_parsed: dict[str, dict] = {}  # the 7 MB file is parsed once per run


def load(ctx, spec):
    """Call load_tokenizer and check the shape of what it returns."""
    got = ctx.solution.load_tokenizer(spec)
    check.that(
        isinstance(got, tuple) and len(got) == 2 and all(callable(f) for f in got),
        "load_tokenizer must return a tuple of two functions, (encode, decode)",
        f"got: {got!r}"[:300],
    )
    return got


def check_encode(encode, cases, what, *, hidden=False):
    """`hidden`: the case is hidden, so its messages must not quote the input."""
    for index, item in enumerate(cases):
        text = item["text"]
        got = encode(text)
        shown = repr(text if len(text) <= 60 else text[:57] + "...")
        label = f"{what}: encode({f'text {index}' if hidden else shown})"
        check.that(
            isinstance(got, list) and all(type(i) is int for i in got),
            f"{label} must return a list of Python ints",
            f"got: {got!r}"[:300],
        )
        check.equal(got, item["ids"], label=label)


def check_decode(decode, cases, what, *, hidden=False):
    for index, item in enumerate(cases):
        ids = item["ids"]
        got = decode(list(ids))
        shown = ids if len(ids) <= 10 else [*ids[:10], "..."]
        label = f"{what}: decode({f'id list {index}' if hidden else shown})"
        check.that(isinstance(got, str), f"{label} must return a str")
        check.equal(got, item["text"], label=label)


def pick(cases, texts):
    """The fixture entries for `texts`, in that order."""
    by_text = {item["text"]: item for item in cases}
    return [by_text[text] for text in texts]


def mini(ctx):
    return ctx.fixture("mini-tokenizer.json"), ctx.fixture("mini-cases.json")


def granite(ctx):
    cases = ctx.fixture("granite-4.0-350m.json")
    check.that(cases["revision"] == REVISION, "the fixture was made from another revision")
    if "spec" not in _parsed:
        path = ctx.asset(ASSET, "tokenizer.json")
        _parsed["spec"] = json.loads(path.read_text(encoding="utf-8"))
    return _parsed["spec"], cases


# ---------------------------------------------------------------------------------
# No download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: a ten-token tokenizer, 'low lower<|eot|>'", public=True, assets=[])
def worked_example(ctx):
    example = ctx.fixture("example.json")
    encode, decode = load(ctx, example["tokenizer"])
    check.equal(encode("low lower<|eot|>"), [8, 9, 6, 10], label="encode('low lower<|eot|>')")
    check_encode(encode, example["encode"], "example")
    check.equal(decode([8, 9, 6, 10]), "low lower<|eot|>", label="decode([8, 9, 6, 10])")
    check_decode(decode, example["decode"], "example")


# Rejects ranked_by_vocab_id, id_from_rank, and leftmost_pair_first: in this file the
# ids are a shuffle, so neither their order nor their value follows the merge list.
@case("a 456-token miniature of Granite's file: English text", public=True, assets=[])
def mini_english(ctx):
    spec, cases = mini(ctx)
    encode, _ = load(ctx, spec)
    texts = [
        "the fox",
        "",
        "The quick brown fox doesn't care.",
        "tokens and tokenizers",
        "It's what we're here for; I'D SAY SO.",
        "aaaaaaaaaa tttttt the thethethe",
    ]
    check_encode(encode, pick(cases["encode"], texts), "miniature")


# Rejects latin1_bytes (bytes 0x80 to 0xA0 occur in every one of these scripts).
@case("the miniature: whitespace runs, newlines, digits, accents, CJK, and emoji",
      public=True, assets=[])  # fmt: skip
def mini_whitespace_unicode(ctx):
    spec, cases = mini(ctx)
    encode, _ = load(ctx, spec)
    texts = [
        " ",
        "hello  world   and    more",
        "   leading and trailing   ",
        "line one\nline two\n\n\nline five",
        "\ttabs\tand\r\nwindows newlines\r\n",
        "2026 and 352379904 and 12 345",
        "naïve café",
        "東京は日本の首都",
        "🙂🚀 emoji 👩‍💻",
    ]
    check_encode(encode, pick(cases["encode"], texts), "miniature")


# Rejects added_after_split and shortest_added_match ("<|end_of_role|>\n" is a token).
@case("the miniature: added tokens in the text, and strings that only look like them",
      public=True, assets=[])  # fmt: skip
def mini_added_tokens(ctx):
    spec, cases = mini(ctx)
    encode, _ = load(ctx, spec)
    texts = [
        "<|start_of_role|>user<|end_of_role|>hello there<|end_of_text|>",
        "<|end_of_text|><|end_of_text|>",
        "text<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "almost special: <|end_of_text |> <|start_of_role| > <end_of_text>",
        "<|start_of_role|><|end_of_role|> the quick <|end_of_text|> brown fox",
        "<|start_of_role|>user<|end_of_role|>\nhi<|end_of_role|> \n<|end_of_role|>\n\n",
    ]
    check_encode(encode, pick(cases["encode"], texts), "miniature")


@case("the miniature: every remaining text", assets=[])
def mini_everything(ctx):
    spec, cases = mini(ctx)
    encode, _ = load(ctx, spec)
    check_encode(encode, cases["encode"], "miniature", hidden=True)


# Rejects decode_per_token (encodings cut inside a character) and decode_skips_specials.
@case("the miniature: decoding, including ids that end in the middle of a character",
      public=True, assets=[])  # fmt: skip
def mini_decode(ctx):
    spec, cases = mini(ctx)
    _, decode = load(ctx, spec)
    check_decode(decode, cases["decode"], "miniature")


@case("decode(encode(text)) is the text", assets=[])
def round_trip(ctx):
    spec, cases = mini(ctx)
    encode, decode = load(ctx, spec)
    for item in cases["encode"]:
        text = item["text"]
        check.equal(decode(encode(text)), text, label="decode(encode(text))")


@case("two tokenizers loaded side by side stay independent; the JSON is not modified",
      assets=[])  # fmt: skip
def independent_instances(ctx):
    spec, cases = mini(ctx)
    example = ctx.fixture("example.json")
    before = copy.deepcopy(spec)
    encode_mini, decode_mini = load(ctx, spec)
    encode_small, decode_small = load(ctx, example["tokenizer"])
    check_encode(
        encode_mini, cases["encode"][:6], "miniature, after loading another tokenizer", hidden=True
    )
    check_encode(encode_small, example["encode"], "example, loaded second", hidden=True)
    check_decode(
        decode_mini, cases["decode"][:6], "miniature, after loading another tokenizer", hidden=True
    )
    check_decode(decode_small, example["decode"], "example, loaded second", hidden=True)
    check_encode(
        encode_mini, cases["encode"][:6], "miniature, encoding the same texts again", hidden=True
    )
    check.that(spec == before, "load_tokenizer or the functions it returned modified the JSON")


# ---------------------------------------------------------------------------------
# The real tokenizer.json of granite-4.0-350m: 100,352 ids, 100,000 merges.
# ---------------------------------------------------------------------------------


@case("Granite 4.0: prose, code, and numbers ('In 2026' is [644, 220, 2366, 21])", public=True)
def granite_text(ctx):
    spec, cases = granite(ctx)
    encode, _ = load(ctx, spec)
    check.equal(encode("In 2026"), [644, 220, 2366, 21], label="Granite: encode('In 2026')")
    check_encode(encode, cases["encode"][:8], "Granite")


@case("Granite 4.0: emoji, CJK, and other scripts", public=True)
def granite_unicode(ctx):
    spec, cases = granite(ctx)
    encode, _ = load(ctx, spec)
    check_encode(encode, cases["encode"][8:11], "Granite")


# Rejects only_special_added_tokens: <think> and <tool_call> are added tokens with
# "special": false. Nothing in the miniature can tell (all its added tokens are special).
@case("Granite 4.0: the chat format, special tokens, and added tokens that are not special")
def granite_added_tokens(ctx):
    spec, cases = granite(ctx)
    encode, _ = load(ctx, spec)
    check_encode(encode, cases["encode"][11:15], "Granite", hidden=True)


@case("Granite 4.0: long inputs and everything else")
def granite_everything(ctx):
    spec, cases = granite(ctx)
    encode, _ = load(ctx, spec)
    check_encode(encode, cases["encode"][15:], "Granite", hidden=True)


@case("Granite 4.0: decoding whole, cut, and random id sequences")
def granite_decode(ctx):
    spec, cases = granite(ctx)
    encode, decode = load(ctx, spec)
    check_decode(decode, cases["decode"], "Granite", hidden=True)
    for item in cases["encode"]:
        text = item["text"]
        check.equal(decode(encode(text)), text, label="Granite: decode(encode(text))")
