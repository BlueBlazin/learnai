"""Regenerate the fixtures of tokenizer-json-load.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/tokenization/tokenizer-json-load/fixtures/make.py

Needs the small house model on disk (``learnai assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is the ``tokenizers`` library; every
expected id and every expected string below comes from it.

- ``mini-tokenizer.json`` is a miniature of Granite's ``tokenizer.json``: the same
  structure and the same pre-tokenizer pattern, a vocabulary of the 256 bytes plus 200
  merges trained with ``tokenizers`` on the small corpus below, and four special
  tokens. Its token ids are then shuffled, so that an id is *not* 256 plus the merge
  rank as it happens to be in Granite's file. ``mini-cases.json`` holds texts with the
  ids ``tokenizers`` gives them under that file, and id lists with their decoded text.
- ``example.json`` is the statement's worked example: a ten-token tokenizer in the same
  structure, with the ids ``tokenizers`` gives the example texts.
- ``granite-4.0-350m.json`` holds the same two lists for the real ``tokenizer.json``.
"""

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from learnai import assets
from learnai.kit.realmodel import SMALL
from learnai.problem import find_root

HERE = Path(__file__).parent

CORPUS = [
    "the quick brown fox jumps over the lazy dog. The dog isn't amused; the fox doesn't care.",
    "tokens, tokenizers, and tokenization: the model only ever sees token ids.",
    "def encode(text):\n    return [vocab[t] for t in bpe(text)]\n",
    "for i in range(10):\n        print(i, i * i)\n\n\nprint('done')",
    "In 2026 the 350M model has 352379904 parameters and 28 layers; 1024 wide, 16 heads.",
    "naïve café déjà vu — señor, où est la bibliothèque? Übergrößenträger!",
    "東京は日本の首都です。日本語のテキストも同じ仕組みで分割されます。",
    "emoji are bytes too: 🙂 🙃 🚀🚀 👩‍💻 and flags 🇯🇵",
    "    indented    text\twith\ttabs   and   runs   of   spaces   \n",
    "user: hello there\nassistant: hello! how can I help you today?\n",
    "It's what we're here for, and they'll say I'd've done the same.",
    "x = {'a': 1, 'b': [2, 3]}; y = x['b'][0] + 40_000 - 3.14159",
]
# The last one extends the one before it: at one position the longest match must win.
MINI_SPECIALS = ["<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>", "<|end_of_role|>\n"]

MINI_TEXTS = [
    "the fox",
    "",
    " ",
    "The quick brown fox doesn't care.",
    "tokens and tokenizers",
    "hello  world   and    more",
    "   leading and trailing   ",
    "line one\nline two\n\n\nline five",
    "\ttabs\tand\r\nwindows newlines\r\n",
    "naïve café",
    "東京は日本の首都",
    "🙂🚀 emoji 👩‍💻",
    "print(i, i * i)",
    "It's what we're here for; I'D SAY SO.",
    "x['b'][0] + 40_000 - 3.14159",
    "2026 and 352379904 and 12 345",
    "<|start_of_role|>user<|end_of_role|>hello there<|end_of_text|>",
    "<|end_of_text|><|end_of_text|>",
    "text<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    "almost special: <|end_of_text |> <|start_of_role| > <end_of_text>",
    "<|start_of_role|><|end_of_role|> the quick <|end_of_text|> brown fox",
    "<|start_of_role|>user<|end_of_role|>\nhi<|end_of_role|> \n<|end_of_role|>\n\n",
    "aaaaaaaaaa tttttt the thethethe",
]

GRANITE_TEXTS = [
    "The capital of France is",
    "",
    "Hello, world! Hello,  world!   Hello,    world!",
    "In 2026, 352379904 parameters; pi is 3.14159 and 1,000,000 is a million.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "    four spaces,\ttab,\n\n\nthree newlines, trailing spaces   ",
    "\r\nWindows\r\nline endings\r\n",
    "It's what we're here for: I'd say they'll've DON'T and CAN'T.",
    "Whetstone sharpens 🗡️ blades; 東京 is in Japan. 👩‍💻🚀🇯🇵",
    "日本語のテキストも同じ仕組みで分割されます。中文也是一样的。한국어도 마찬가지입니다.",
    "naïve café, señor — Übergrößenträger; Ελληνικά, кириллица, עברית, العربية",
    "<|start_of_role|>user<|end_of_role|>What is 2+2?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    "<|end_of_text|><|end_of_text|> <|pad|>",
    '<think>Let me check.</think><tool_call>{"name": "add"}</tool_call>',
    "almost: <|end_of_text |> <think > <|unused_1|> <|unused_100|> <tools></tools>",
    "x = {'a': 1, 'b': [2, 3]}; y = x['b'][0] + 40_000 - 0x1F  # comment",
    "https://example.com/path?query=1&other=two#fragment user@example.com",
    "supercalifragilisticexpialidocious antidisestablishmentarianism " * 3,
    "The tokenizer is the first stage of every inference request, and the model only works "
    "on the exact ids it saw in training. " * 8,
]


def decode_cases(tok: Tokenizer, texts: list[str], n_ids: int, seed: int) -> list[dict]:
    """Id lists and what the library decodes them to: whole encodings, encodings cut
    in the middle of multi-byte characters, and random ids."""
    rng = np.random.default_rng(seed)
    cases = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append(ids)
        if len(ids) > 3:
            cases.append(ids[: len(ids) // 2])
            cases.append(ids[1::2])
    for _ in range(6):
        cases.append([int(i) for i in rng.integers(0, n_ids, size=24)])
    cases.append([])
    return [{"ids": ids, "text": tok.decode(ids, skip_special_tokens=False)} for ids in cases]


def write(name: str, payload: dict) -> None:
    target = HERE / name
    target.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8"
    )
    print(f"wrote {name} ({target.stat().st_size} bytes)")


def mini(granite_spec: dict) -> None:
    spec = json.loads(Tokenizer(models.BPE()).to_str())
    spec["pre_tokenizer"] = granite_spec["pre_tokenizer"]  # the same Split + ByteLevel
    spec["decoder"] = granite_spec["decoder"]
    tok = Tokenizer.from_str(json.dumps(spec))
    trainer = trainers.BpeTrainer(
        vocab_size=256 + 200,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS * 3, trainer)
    spec = json.loads(tok.to_str())
    # Shuffle the ids of the whole vocabulary: ids stop being "256 + merge rank", and
    # the order of ids stops being the order of the merges.
    vocab = spec["model"]["vocab"]
    order = np.random.default_rng(2).permutation(len(vocab))
    spec["model"]["vocab"] = {token: int(order[i]) for i, token in enumerate(vocab)}
    spec["added_tokens"] = [
        {"id": len(vocab) + i, "content": content, "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True}
        for i, content in enumerate(MINI_SPECIALS)
    ]  # fmt: skip
    tok = Tokenizer.from_str(json.dumps(spec))
    spec = json.loads(tok.to_str())  # as the library writes it
    write("mini-tokenizer.json", spec)
    encode = [
        {"text": text, "ids": tok.encode(text, add_special_tokens=False).ids} for text in MINI_TEXTS
    ]
    n_ids = len(vocab) + len(MINI_SPECIALS)
    write("mini-cases.json", {"encode": encode, "decode": decode_cases(tok, MINI_TEXTS, n_ids, 3)})
    example(spec)


def example(mini_spec: dict) -> None:
    spec = json.loads(json.dumps(mini_spec))
    spec["model"]["vocab"] = {
        "e": 0, "l": 1, "o": 2, "r": 3, "w": 4, "Ġ": 5, "er": 6, "lo": 7, "low": 8, "Ġlow": 9,
    }  # fmt: skip
    spec["model"]["merges"] = [["l", "o"], ["lo", "w"], ["e", "r"], ["Ġ", "low"]]
    spec["added_tokens"] = [dict(mini_spec["added_tokens"][0], id=10, content="<|eot|>")]
    tok = Tokenizer.from_str(json.dumps(spec))
    texts = ["low lower<|eot|>", "lower", " low", "<|eot|>low<|eot|> lowlow", "ole  role"]
    encode = [
        {"text": text, "ids": tok.encode(text, add_special_tokens=False).ids} for text in texts
    ]
    decode = [
        {"ids": ids, "text": tok.decode(ids, skip_special_tokens=False)}
        for ids in ([8, 9, 6, 10], [10, 5, 5, 7, 4], [])
    ]
    write("example.json", {"tokenizer": spec, "encode": encode, "decode": decode})


def granite(directory: Path) -> None:
    tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
    encode = [
        {"text": text, "ids": tok.encode(text, add_special_tokens=False).ids}
        for text in GRANITE_TEXTS
    ]
    payload = {
        "revision": SMALL.revision,
        "encode": encode,
        "decode": decode_cases(tok, GRANITE_TEXTS[:16], 100352, 4),
    }
    write("granite-4.0-350m.json", payload)


if __name__ == "__main__":
    model_dir = assets.resolve(SMALL, assets.models_dir(find_root()))
    mini(json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8")))
    granite(model_dir)
