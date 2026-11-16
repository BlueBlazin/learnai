"""Regenerate the fixtures of real-model-chat-turn.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/tokenization/real-model-chat-turn/fixtures/make.py

Needs the small house model on disk (``learnai assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementations are ``transformers`` (the model's own
Jinja chat template through ``apply_chat_template``, and greedy ``generate`` in
float32 on the CPU) and ``tokenizers``.

- ``mini-tokenizer.json``: a miniature of Granite's ``tokenizer.json`` for the cases
  that need no download: the same pre-tokenizer, the 256 bytes plus 120 merges trained
  on the small corpus below, and six added tokens with Granite's names (but not its
  ids). A character outside the corpus, such as an emoji, is one token per byte.
- ``mini-cases.json``: conversations rendered by the real template, tokenized with
  the miniature tokenizer, each with the tokens a scripted model emits, the tokens
  that count as generated, and the text pieces a streaming client must receive.
- ``granite-4.0-350m.json``: conversations with the prompt ids of
  ``apply_chat_template`` and the greedy continuation of ``generate``; pieces as above.

The pieces come from Python's incremental UTF-8 decoder over the token bytes; their
concatenation is asserted to equal what the libraries decode the whole reply to.
"""

import codecs
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, AutoTokenizer

from learnai import assets
from learnai.kit import ByteLevelBPE
from learnai.kit.realmodel import SMALL
from learnai.problem import find_root

HERE = Path(__file__).parent
STOP = "<|end_of_text|>"

CORPUS = [
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe.",
    "system user assistant: Hello there! How can I help you today? Hi! Thanks.",
    "What is the capital of France? The capital of France is Paris. And of Japan? Tokyo.",
    "Name a color. Blue. Another one. Red. Answer in one word, please.",
    "Be brief. Now answer in German. Wie geht es dir? Gut, danke.",
    "the model reads the prompt once and then writes one token at a time",
]
#: (content, special) in id order after the 376 ordinary tokens.
MINI_ADDED = [
    ("<|pad|>", True),
    ("<|start_of_role|>", True),
    ("<|end_of_role|>", True),
    (STOP, True),
    ("<|unk|>", True),
    ("<think>", False),
]


def raw(*values: int) -> dict:
    """Script item: tokens for these single bytes, one token per byte."""
    return {"bytes": list(values)}


EMOJI = raw(0xF0, 0x9F, 0x99, 0x82)  # the four bytes of U+1F642
#: name, messages, script, max_new_tokens. A script item is text (encoded with the
#: miniature tokenizer), an added token (by its content), or raw bytes.
MINI_CASES = [
    ("example", [{"role": "user", "content": "Hi!"}],
     ["Hello", "!", " ", EMOJI, STOP, " never read"], 16),
    ("system_first",
     [{"role": "system", "content": "Be brief."},
      {"role": "user", "content": "Name a color."},
      {"role": "assistant", "content": "Blue."},
      {"role": "user", "content": "Another one."}],
     ["Red.", STOP], 16),
    ("empty_system",
     [{"role": "system", "content": ""},
      {"role": "user", "content": "  What is the capital\nof France?\n"},
      {"role": "assistant", "content": ""},
      {"role": "system", "content": "Now answer in German."},
      {"role": "user", "content": "And of Japan? "}],
     ["Tokyo", ".", STOP], 16),
    ("assistant_first", [{"role": "assistant", "content": "Hello there!"},
                         {"role": "user", "content": "Thanks."}],
     [" How can I help", "?", STOP], 16),
    ("cut_in_character", [{"role": "user", "content": "Hi!"}],
     ["Hi", EMOJI, "!", STOP], 3),
    ("limit_reached", [{"role": "user", "content": "Name a color."}],
     ["Blue", " Red", " Blue", " Red", " Blue", STOP], 4),
    ("limit_is_reply_length", [{"role": "user", "content": "Name a color."}],
     ["Blue", ".", STOP], 2),
    ("one_token", [{"role": "user", "content": "Name a color."}],
     ["Blue", ".", STOP], 1),
    ("special_tokens", [{"role": "user", "content": "Hi!"}],
     ["<think>", "Hello", "<|pad|>", "<|start_of_role|>", " there", "<|unk|>", STOP], 16),
    ("immediate_stop", [{"role": "user", "content": "Be brief."}], [STOP, "Hello"], 16),
    ("invalid_bytes", [{"role": "user", "content": "Hi!"}],
     [raw(0x99), "a", raw(0xE2, 0x82), "b", raw(0xC3), raw(0xA9), raw(0xFF), EMOJI, raw(0xF0, 0x9F),
      STOP], 32),
    ("multibyte_text", [{"role": "user", "content": "Wie geht es dir?"}],
     ["Gut", ", ", raw(0xE6, 0x84, 0x9F), raw(0xE8), raw(0xAC, 0x9D), " danke", ".", STOP], 32),
]  # fmt: skip

REAL_CASES = [
    ("three_emoji",
     [{"role": "user", "content": "Reply with three different emoji and nothing else."}], 16),
    ("translator",
     [{"role": "system", "content": "You are a translator. Reply only with the translation."},
      {"role": "user", "content": "Translate to Japanese: Thank you very much."}], 16),
    ("second_turn",
     [{"role": "user", "content": "Name a color."},
      {"role": "assistant", "content": "Blue."},
      {"role": "user", "content": "Another one."}], 16),
    ("two_system_messages",
     [{"role": "system", "content": "Be brief."},
      {"role": "user", "content": "Hi!"},
      {"role": "system", "content": "Now answer in German."},
      {"role": "user", "content": "How are you?"}], 16),
    ("cut_in_emoji",
     [{"role": "user", "content": "Reply with three different emoji and nothing else."}], 5),
]  # fmt: skip
#: Greedy tokens must win by at least this much, so float noise cannot change them.
MARGIN = 0.25


def pieces_of(kit: ByteLevelBPE, generated: list[int]) -> list[str]:
    """The non-empty texts a streaming client receives, one decoder step per token."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    special = set(kit.specials.values())
    out = [decoder.decode(b"" if t in special else kit.token_bytes(t)) for t in generated]
    out.append(decoder.decode(b"", final=True))
    return [piece for piece in out if piece]


def write(name: str, cases: list[dict], **extra: object) -> None:
    """One case per line, so that a diff of the fixture is readable."""
    lines = ",\n".join("  " + json.dumps(case, ensure_ascii=False) for case in cases)
    head = "".join(f'"{key}": {json.dumps(value)},\n ' for key, value in extra.items())
    target = HERE / name
    target.write_text(f'{{{head}"cases": [\n{lines}\n]}}\n', "utf-8")
    print(f"wrote {name} ({target.stat().st_size} bytes)")


def mini_tokenizer(granite_spec: dict) -> Tokenizer:
    spec = json.loads(Tokenizer(models.BPE()).to_str())
    spec["pre_tokenizer"] = granite_spec["pre_tokenizer"]
    spec["decoder"] = granite_spec["decoder"]
    tok = Tokenizer.from_str(json.dumps(spec))
    trainer = trainers.BpeTrainer(
        vocab_size=256 + 120,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS * 3, trainer)
    spec = json.loads(tok.to_str())
    n = len(spec["model"]["vocab"])
    spec["added_tokens"] = [
        {"id": n + i, "content": content, "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": special}
        for i, (content, special) in enumerate(MINI_ADDED)
    ]  # fmt: skip
    tok = Tokenizer.from_str(json.dumps(spec))
    (HERE / "mini-tokenizer.json").write_text(
        json.dumps(json.loads(tok.to_str()), ensure_ascii=False, separators=(",", ":")) + "\n",
        "utf-8",
    )
    print(f"wrote mini-tokenizer.json ({(HERE / 'mini-tokenizer.json').stat().st_size} bytes)")
    return tok


def mini(real: AutoTokenizer, granite_spec: dict) -> None:
    tok = mini_tokenizer(granite_spec)
    kit = ByteLevelBPE.from_file(HERE / "mini-tokenizer.json")
    stop = tok.token_to_id(STOP)
    # The token of every single byte: what a character outside the corpus falls apart into.
    byte_token = {kit.token_bytes(i)[0]: i for i in kit.vocab.values() if len(kit.token_bytes(i)) == 1}  # fmt: skip
    assert len(byte_token) == 256
    cases = []
    for name, messages, script_items, limit in MINI_CASES:
        # The model's own template, rendered by transformers; ids by `tokenizers`.
        text = real.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(text, add_special_tokens=False).ids
        assert kit.encode(text) == prompt_ids
        script: list[int] = []
        for item in script_items:
            if isinstance(item, dict):
                script.extend(byte_token[b] for b in item["bytes"])
            else:
                script.extend(tok.encode(item, add_special_tokens=False).ids)
        generated: list[int] = []
        for token in script:
            if token == stop or len(generated) == limit:
                break
            generated.append(token)
        stopped = len(generated) < limit  # ended by the stop token, not by the limit
        pieces = pieces_of(kit, generated)
        whole = tok.decode(generated, skip_special_tokens=True)
        assert "".join(pieces) == whole, (name, pieces, whole)
        cases.append(
            {
                "name": name,
                "messages": messages,
                "max_new_tokens": limit,
                "prompt_text": text,
                "prompt_ids": prompt_ids,
                "script": script,
                "generated": generated,
                "stopped": stopped,
                "pieces": pieces,
            }
        )
        print(f"{name}: {len(prompt_ids)} prompt tokens, {len(generated)} generated, {pieces}")
    write("mini-cases.json", cases)


def granite(real: AutoTokenizer, directory: Path) -> None:
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    kit = ByteLevelBPE.from_file(directory / "tokenizer.json")
    stop = kit.specials[STOP]
    cases = []
    for name, messages, limit in REAL_CASES:
        text = real.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = real.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids["input_ids"])
        assert kit.encode(text) == prompt_ids
        with torch.no_grad():
            out = model.generate(
                torch.tensor([prompt_ids]), max_new_tokens=limit, do_sample=False,
                output_scores=True, return_dict_in_generate=True, pad_token_id=stop,
            )  # fmt: skip
        new = out.sequences[0, len(prompt_ids) :].tolist()
        margins = [float(s[0].topk(2).values[0] - s[0].topk(2).values[1]) for s in out.scores]
        assert min(margins) > MARGIN, (name, margins)
        stopped = new[-1] == stop
        generated = new[:-1] if stopped else new
        assert stop not in generated and (stopped or len(generated) == limit)
        pieces = pieces_of(kit, generated)
        whole = real.decode(generated, skip_special_tokens=True)
        assert "".join(pieces) == whole, (name, pieces, whole)
        cases.append(
            {
                "name": name,
                "messages": messages,
                "max_new_tokens": limit,
                "prompt_text": text,
                "prompt_ids": prompt_ids,
                "generated": generated,
                "stopped": stopped,
                "pieces": pieces,
            }
        )
        print(f"{name}: {len(prompt_ids)} prompt tokens, {len(generated)} generated, {pieces}, "
              f"smallest margin {min(margins):.2f}")  # fmt: skip
    write("granite-4.0-350m.json", cases, revision=SMALL.revision)


if __name__ == "__main__":
    model_dir = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    mini(tokenizer, json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8")))
    granite(tokenizer, model_dir)
