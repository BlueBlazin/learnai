"""Regenerate the fixtures of capstone-real-model-runtime.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/capstones/capstone-real-model-runtime/fixtures/make.py

Needs the small house model on disk (``learnai assets fetch granite-4.0-350m``) and the
``models`` extra. The oracle is the kit: ``RealLM`` (the numpy forward pass, with its
own cache) for logits, keys, and values, and ``ByteLevelBPE`` for token ids and token
bytes. Three things are cross-checked against the libraries while the fixtures are
made: token ids against ``tokenizers``, chat prompts against the model's own Jinja
template rendered by ``transformers``, and the real model's logits against
``transformers`` in float32.

- ``tiny/``: a complete model directory for the cases that need no download.
  ``tokenizer.json`` is the 382-token miniature of Granite's tokenizer from
  ``real-model-chat-turn`` (same pre-tokenizer and special-token names, other ids);
  ``config.json`` describes a two-layer model in Granite's layout with *other*
  multipliers; ``model.safetensors`` holds its random weights in bfloat16 (83 kB).
- ``tiny.npz`` and ``tiny.json``: what the tiny model computes: token ids, chat
  prompts, logits, the cache after a prefill, decode steps, sampled generations, and
  streamed replies with stop strings.
- ``granite-4.0-350m.npz`` and ``granite-4.0-350m.json``: the same for the real
  model, with logits at 64 fixed vocabulary columns and a few slices of the cache.

Sampling is exact only if float noise cannot change a token. Every uniform number
stored here was accepted after checking that the sampled token survives 32 random
perturbations of the logits by up to 3e-3 (thirty times the difference between
frameworks) and a shift of the uniform by 1e-4.
"""

import codecs
import json
from pathlib import Path

import numpy as np

from learnai import assets
from learnai.kit import RealModel
from learnai.kit.realmodel import SMALL
from learnai.problem import find_root

HERE = Path(__file__).parent
TINY_DIR = HERE / "tiny"
MINI_TOKENIZER = HERE.parents[2] / "tokenization/real-model-chat-turn/fixtures/mini-tokenizer.json"
COLUMNS = np.sort(np.random.default_rng(150).choice(100352, size=64, replace=False))

TINY_CONFIG = {
    "architectures": ["GraniteMoeHybridForCausalLM"],
    "model_type": "granitemoehybrid",
    "layer_types": ["attention", "attention"],
    "num_local_experts": 0,
    "vocab_size": 382,
    "hidden_size": 48,
    "intermediate_size": 32,
    "shared_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_multiplier": 0.125,
    "embedding_multiplier": 6.0,
    "residual_multiplier": 0.4,
    "logits_scaling": 0.5,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": True,
    "max_position_embeddings": 4096,
    "bos_token_id": 379,
    "eos_token_id": 379,
    "pad_token_id": 376,
    "torch_dtype": "bfloat16",
}

GREEDY = {
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "temperature": 1.0,
}
PLAIN = {**GREEDY, "top_k": 0}
CHAIN = {
    "repetition_penalty": 1.15,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.1,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.02,
    "temperature": 0.8,
}
HOT = {**PLAIN, "top_k": 12, "temperature": 1.6, "repetition_penalty": 1.3}
NUCLEUS = {**PLAIN, "top_p": 0.6, "min_p": 0.1, "frequency_penalty": 0.5, "presence_penalty": 0.4}


# -- writing a safetensors file ------------------------------------------------------------


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest, ties to even) and return the 16-bit patterns."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """All tensors as BF16, in name order, the header padded to a multiple of 8."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    chunks, offset = [], 0
    for name in sorted(tensors):
        data = to_bf16(tensors[name]).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(len(text).to_bytes(8, "little") + text + b"".join(chunks))


def tiny_weights(seed: int) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout.

    The branch matrices are large and the embedding is small, so that attention and
    the MLP move the stream: otherwise a tied head only ever repeats the last token.
    """
    rng = np.random.default_rng(seed)
    cfg = TINY_CONFIG
    d, f, vocab = cfg["hidden_size"], cfg["shared_intermediate_size"], cfg["vocab_size"]
    kv = d * cfg["num_key_value_heads"] // cfg["num_attention_heads"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, d).astype(np.float32)

    weights = {"model.embed_tokens.weight": normal(vocab, d, 0.15)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(d, d, 0.4)
        weights[p + "self_attn.k_proj.weight"] = normal(kv, d, 0.4)
        weights[p + "self_attn.v_proj.weight"] = normal(kv, d, 0.4)
        weights[p + "self_attn.o_proj.weight"] = normal(d, d, 0.4)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.3)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.4)
    weights["model.norm.weight"] = gain()
    return weights


# -- the sampler chain, in float64 ---------------------------------------------------------


def softmax(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    z = np.where(finite, x - x[finite].max(), -np.inf)
    e = np.exp(z)
    return e / e.sum()


def chain_probs(logits: np.ndarray, s: dict, history: list[int]) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).copy()
    size = x.shape[0]
    counts = np.bincount(np.asarray(history, dtype=np.int64), minlength=size)
    seen = counts > 0
    theta = s["repetition_penalty"]
    x = np.where(seen, np.where(x > 0, x / theta, x * theta), x)
    x = x - counts * s["frequency_penalty"] - seen * s["presence_penalty"]
    index = np.arange(size)
    if 0 < s["top_k"] < size:
        order = np.lexsort((index, -x))  # decreasing logit, ties by index
        x[order[s["top_k"] :]] = -np.inf
    if s["top_p"] < 1.0:
        p = softmax(x)
        order = np.lexsort((index, -p))
        before = np.cumsum(p[order]) - p[order]
        x[order[before >= s["top_p"]]] = -np.inf
    if s["min_p"] > 0.0:
        p = softmax(x)
        x[p < s["min_p"] * p.max()] = -np.inf
    return softmax(x / s["temperature"])


def draw(probs: np.ndarray, u: float) -> int:
    cdf = np.cumsum(probs)
    return int((cdf <= u * cdf[-1]).sum())


def robust_uniform(rng, logits: np.ndarray, s: dict, history: list[int]) -> tuple[float, int]:
    """A uniform number whose token no float noise can change, and that token."""
    noise = np.random.default_rng(7)
    for _ in range(400):
        u = float(np.float32(rng.uniform(0.02, 0.98)))
        token = draw(chain_probs(logits, s, history), u)
        stable = all(
            draw(chain_probs(logits + noise.uniform(-3e-3, 3e-3, logits.shape), s, history), v)
            == token
            for v in (u, u - 1e-4, u + 1e-4, *[u] * 30)
        )
        if stable:
            return u, token
    raise RuntimeError("no robust uniform found; change the sampler settings or the prompt")


def sample_run(lm, prompt: list[int], n: int, s: dict, seed: int, eos: int) -> dict:
    """Generate up to n tokens with the chain; the uniforms are chosen to be robust."""
    rng = np.random.default_rng(seed)
    logits, cache = lm(np.array(prompt))
    row = np.asarray(logits)[-1]
    first = row.copy()
    tokens, uniforms = [], []
    for step in range(n):
        u, token = robust_uniform(rng, row, s, tokens)
        uniforms.append(u)
        if token == eos:
            break
        tokens.append(token)
        if step + 1 < n:
            logits, cache = lm(np.array([token]), cache)
            row = np.asarray(logits)[-1]
    # Uniforms for the steps that were not reached are still part of the input.
    uniforms += [float(np.float32(rng.uniform(0.02, 0.98))) for _ in range(n - len(uniforms))]
    return {"tokens": tokens, "uniforms": uniforms, "first_logits": first}


# -- text: incremental UTF-8 and stop strings ------------------------------------------------


def visible(text: str, stops: list[str], final: bool) -> tuple[int, bool]:
    ends = [(text.find(s) + len(s), text.find(s)) for s in stops if s in text]
    if ends:
        return min(ends)[1], True
    if final:
        return len(text), False
    held = 0
    for s in stops:
        for size in range(1, len(s)):
            if text.endswith(s[:size]):
                held = max(held, size)
    return len(text) - held, False


def pieces_of(tokenizer, tokens: list[int], stops: list[str]) -> tuple[list[str], int]:
    """The strings a streaming client receives, and how many tokens were consumed."""
    special = set(tokenizer.specials.values())
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text, shown, pieces = "", 0, []
    for used, token in enumerate(tokens, start=1):
        text += decoder.decode(b"" if token in special else tokenizer.token_bytes(token))
        end, stopped = visible(text, stops, False)
        if end > shown:
            pieces.append(text[shown:end])
            shown = end
        if stopped:
            return pieces, used
    text += decoder.decode(b"", final=True)
    end, _ = visible(text, stops, True)
    if end > shown:
        pieces.append(text[shown:end])
    return pieces, len(tokens)


def render(messages: list[dict]) -> str:
    """Granite's chat layout, written out; asserted against the Jinja template below."""
    default = (
        "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
    )
    system, rest = default, messages
    if messages[0]["role"] == "system":
        system, rest = messages[0]["content"] or default, messages[1:]
    turns = [("system", system)] + [(m["role"], m["content"]) for m in rest]
    text = "".join(f"<|start_of_role|>{r}<|end_of_role|>{c}<|end_of_text|>\n" for r, c in turns)
    return text + "<|start_of_role|>assistant<|end_of_role|>"


CHATS = [
    [{"role": "user", "content": "Hi!"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Another one."},
    ],
    [
        {"role": "system", "content": ""},
        {"role": "user", "content": "  What is the capital\nof France?\n"},
        {"role": "assistant", "content": ""},
        {"role": "system", "content": "Answer in one word, please."},
        {"role": "user", "content": "And of Japan?"},
    ],
]
MINI_TEXTS = [
    "Hello there! How can I help you today?",
    "the model reads the prompt once",
    "In 2026, 12345 tokens.\n\n  Two spaces",
    "café \U0001f642 <think>ok<|end_of_text|>",
    "",
]
REAL_TEXTS = [
    "Hello, world!",
    "In 2026",
    "def f(x):\n    return x ** 2  # squared\n",
    "naïve café 日本語 \U0001f642",
    "<|start_of_role|>user<|end_of_role|><think>1,234,567 tokens</think>",
    "",
]


def template_text(reference, messages: list[dict]) -> str:
    text = reference.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert text == render(messages), (text, render(messages))
    return text


def stop_candidates(tokenizer, tokens: list[int]) -> dict[str, list[str]]:
    """Stop strings taken from the reply itself, so that they really occur."""
    text = "".join(pieces_of(tokenizer, tokens, [])[0])
    middle = len(text) // 2
    spanning = text[middle : middle + 3]  # three characters: more than one token, usually
    early = text[2:4]
    # A stop whose first two characters occur early without the third: held, then released.
    decoy = text[1:3] + "\x00"
    return {
        "none": [],
        "spanning": [spanning],
        "decoy": [decoy, spanning],
        "two": [spanning, early],
        "absent": ["\x01\x02", text[-2:] + "\x03"],
    }


def tiny(reference) -> None:
    TINY_DIR.mkdir(exist_ok=True)
    tokenizer_json = json.loads(MINI_TOKENIZER.read_text("utf-8"))
    (TINY_DIR / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    (TINY_DIR / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n", "utf-8")
    write_safetensors(TINY_DIR / "model.safetensors", tiny_weights(155))

    model = RealModel(TINY_DIR)
    lm, tokenizer = model.lm(), model.tokenizer
    eos = TINY_CONFIG["eos_token_id"]
    assert tokenizer.specials["<|end_of_text|>"] == eos
    from tokenizers import Tokenizer as LibraryTokenizer

    library = LibraryTokenizer.from_str(json.dumps(tokenizer_json))
    arrays: dict[str, np.ndarray] = {}
    out: dict[str, object] = {"texts": [], "chats": [], "sampler": [], "generate": [], "stream": []}

    for text in MINI_TEXTS:
        ids = tokenizer.encode(text)
        assert ids == library.encode(text, add_special_tokens=False).ids, text
        out["texts"].append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    for messages in CHATS:
        text = template_text(reference, messages)
        out["chats"].append({"messages": messages, "text": text, "ids": tokenizer.encode(text)})

    # Forward passes: every position of two prompts.
    prompts = [tokenizer.encode("Hello there! How can I help you today?"), [41], [7] * 8]
    for index, prompt in enumerate(prompts):
        logits, _ = lm(np.array(prompt))
        arrays[f"forward_ids_{index}"] = np.array(prompt, dtype=np.int32)
        arrays[f"forward_logits_{index}"] = np.asarray(logits, dtype=np.float32)

    # The cache: after a prefill, and decode steps along the greedy tokens; then a branch.
    prompt = out["chats"][0]["ids"]
    logits, cache = lm(np.array(prompt))
    arrays["cache_prompt"] = np.array(prompt, dtype=np.int32)
    arrays["cache_prefill_logits"] = np.asarray(logits, dtype=np.float32)[-1]
    arrays["cache_keys"] = np.stack([k[0] for k in cache.keys])  # (layers, kv heads, P, hd)
    arrays["cache_values"] = np.stack([v[0] for v in cache.values])
    tokens, rows = [], []
    row = arrays["cache_prefill_logits"]
    start = cache
    for _ in range(6):
        ranked = np.sort(row)
        assert ranked[-1] - ranked[-2] > 0.02, "greedy margin too small"
        tokens.append(int(row.argmax()))
        logits, cache = lm(np.array([tokens[-1]]), cache)
        row = np.asarray(logits, dtype=np.float32)[-1]
        rows.append(row)
    arrays["cache_tokens"] = np.array(tokens, dtype=np.int32)
    arrays["cache_step_logits"] = np.stack(rows)
    arrays["cache_final_keys"] = np.stack([k[0] for k in cache.keys])
    arrays["cache_final_values"] = np.stack([v[0] for v in cache.values])
    other = int(np.argsort(-arrays["cache_prefill_logits"])[1])
    logits, _ = lm(np.array([other]), start)
    arrays["cache_other"] = np.array(other, dtype=np.int32)
    arrays["cache_branch_logits"] = np.asarray(logits, dtype=np.float32)[-1]

    # The sampler chain alone, on small rows.
    rng = np.random.default_rng(151)
    for number, s in enumerate([GREEDY, PLAIN, CHAIN, HOT, NUCLEUS, CHAIN, HOT, NUCLEUS] * 2):
        size = 24
        row = (rng.standard_normal(size) * 2.5).astype(np.float32)
        if number % 3 == 2:
            row[rng.choice(size, 5, replace=False)] = -np.inf  # already masked
        history = [int(t) for t in rng.integers(0, size, int(rng.integers(0, 9)))]
        if number >= 8:
            s = {**s, "top_k": min(s["top_k"], 6) if s["top_k"] else 0}
        u, token = robust_uniform(rng, row, s, history)
        assert np.isfinite(row[token])
        out["sampler"].append(
            {"logits": [float(v) for v in row], "sampler": s, "history": history, "u": u,
             "token": token}
        )  # fmt: skip

    # Sampled generations. `eos` overrides config["eos_token_id"] to end a run early.
    prompt = out["chats"][1]["ids"]
    plan = [("greedy", GREEDY, 10, 1), ("plain", PLAIN, 12, 2), ("chain", CHAIN, 12, 3),
            ("hot", HOT, 12, 4), ("nucleus", NUCLEUS, 12, 5)]  # fmt: skip
    for name, s, n, seed in plan:
        run = sample_run(lm, prompt, n, s, seed, eos)
        assert len(run["tokens"]) == n
        out["generate"].append(
            {"name": name, "prompt": prompt, "sampler": s, "max_new_tokens": n, "eos": eos,
             "uniforms": run["uniforms"], "tokens": run["tokens"]}
        )  # fmt: skip
        print(f"tiny generate {name}: {run['tokens']}")
    # The same chain run with its fifth token declared end-of-sequence.
    chain = out["generate"][2]
    stop_token = chain["tokens"][4]
    assert stop_token not in chain["tokens"][:4]
    out["generate"].append({**chain, "name": "chain_eos", "eos": stop_token,
                            "tokens": chain["tokens"][:4]})  # fmt: skip
    out["generate"].append({**chain, "name": "chain_one", "max_new_tokens": 1,
                            "tokens": chain["tokens"][:1]})  # fmt: skip

    # Streams: chats, sampled, cut by stop strings taken from their own text.
    seeds = iter(range(100, 400))
    for chat, s, n in [(0, HOT, 20), (1, CHAIN, 16), (1, NUCLEUS, 16)]:
        prompt = out["chats"][chat]["ids"]
        # Look for a reply with a special token and an unfinished character in it.
        for seed in seeds:
            run = sample_run(lm, prompt, n, s, seed, eos)
            raw = b"".join(tokenizer.token_bytes(t) for t in run["tokens"])
            specials = set(tokenizer.specials.values()) & set(run["tokens"])
            if (
                len(run["tokens"]) == n
                and "\ufffd" in raw.decode("utf-8", "replace")
                and (specials or chat != 0)
            ):
                break
        for kind, stops in stop_candidates(tokenizer, run["tokens"]).items():
            pieces, used = pieces_of(tokenizer, run["tokens"], stops)
            out["stream"].append(
                {"name": f"chat{chat}_{kind}", "messages": CHATS[chat], "sampler": s,
                 "max_new_tokens": n, "eos": eos, "uniforms": run["uniforms"], "stop": stops,
                 "tokens": run["tokens"], "used": used, "pieces": pieces}
            )  # fmt: skip
            print(f"tiny stream chat{chat}_{kind}: used {used}/{n}, {pieces!r}"[:150])
        # Cut by the limit in the middle of the reply, and ended by end-of-sequence.
        short = n // 2
        pieces, used = pieces_of(tokenizer, run["tokens"][:short], [])
        out["stream"].append(
            {"name": f"chat{chat}_limit", "messages": CHATS[chat], "sampler": s,
             "max_new_tokens": short, "eos": eos, "uniforms": run["uniforms"], "stop": [],
             "tokens": run["tokens"][:short], "used": used, "pieces": pieces}
        )  # fmt: skip
        stop_token = run["tokens"][short]
        if stop_token not in run["tokens"][:short]:
            out["stream"].append(
                {"name": f"chat{chat}_eos", "messages": CHATS[chat], "sampler": s,
                 "max_new_tokens": n, "eos": stop_token, "uniforms": run["uniforms"],
                 "stop": [], "tokens": run["tokens"][:short], "used": used, "pieces": pieces}
            )  # fmt: skip

    # Stop strings in awkward places, on the first reply.
    base = out["stream"][0]
    text = "".join(base["pieces"])
    inside = next(p for p in base["pieces"][3:] if len(p) >= 3 and p.isascii())
    m = len(text) // 3
    out["stream_edges"] = []
    for stops in (
        [text[0]],  # the first character: nothing is ever shown
        [inside[1:3]],  # inside one token: the token's first character is still shown
        [text[m : m + 4], text[m + 1 : m + 3]],  # a stop inside a stop: the inner ends first
        [text[m : m + 3], text[m + 1 : m + 5]],  # overlapping: the first to end
        [text[m + 2 : m + 5], text[m : m + 5]],  # the same end: the longer one
        [text[m : m + 2] + "\x00", text[m + 1 : m + 4]],  # a held prefix overtaken by a match
        [text[-3:]],  # completed by the last token
        [text[:2] + "\x00"],  # held from the first token, released later
    ):
        pieces, used = pieces_of(tokenizer, base["tokens"], stops)
        out["stream_edges"].append({**base, "name": "edge", "stop": stops, "used": used,
                                    "pieces": pieces})  # fmt: skip
        print(f"tiny edge {stops!r}: used {used}, {pieces!r}"[:150])

    np.savez_compressed(HERE / "tiny.npz", **arrays)
    (HERE / "tiny.json").write_text(json.dumps(out, ensure_ascii=True, indent=1) + "\n", "utf-8")
    for name in ("tiny.npz", "tiny.json", "tiny/model.safetensors", "tiny/tokenizer.json"):
        print(f"wrote {name} ({(HERE / name).stat().st_size} bytes)")


def granite(directory: Path, reference) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    model = RealModel(directory)
    lm, tokenizer = model.lm(), model.tokenizer
    eos = model.config_json["eos_token_id"]
    arrays: dict[str, np.ndarray] = {"columns": COLUMNS, "revision": np.array(SMALL.revision)}
    out: dict[str, object] = {"revision": SMALL.revision, "texts": [], "chats": []}

    for text in REAL_TEXTS:
        ids = tokenizer.encode(text)
        assert ids == reference(text, add_special_tokens=False).input_ids, text
        out["texts"].append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    for messages in CHATS:
        text = template_text(reference, messages)
        ids = tokenizer.encode(text)
        assert ids == reference(text, add_special_tokens=False).input_ids
        out["chats"].append({"messages": messages, "text": text, "ids": ids})

    # One forward pass over 12 tokens, cross-checked against transformers.
    prompt = tokenizer.encode("A B C D E F G H I J K L")
    logits = np.asarray(lm(np.array(prompt))[0], dtype=np.float32)
    library = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = library(torch.tensor([prompt])).logits[0].numpy()
    print(f"forward: max |kit - transformers| = {np.abs(logits - theirs).max():.2e}")
    assert np.abs(logits - theirs).max() < 1e-3
    del library
    arrays["forward_ids"] = np.array(prompt, dtype=np.int32)
    arrays["forward_logits"] = logits[:, COLUMNS]
    arrays["forward_argmax"] = logits.argmax(-1).astype(np.int32)

    # A sampled reply to the first chat; slices of the cache after its prefill.
    prompt = out["chats"][0]["ids"]
    run = sample_run(lm, prompt, 10, CHAIN, 21, eos)
    _, cache = lm(np.array(prompt))
    arrays["cache_prefill_logits"] = run["first_logits"][COLUMNS]
    arrays["cache_layers"] = np.array([0, 13, 27], dtype=np.int32)
    arrays["cache_keys"] = np.stack([cache.keys[i][0][:, :, :8] for i in (0, 13, 27)])
    arrays["cache_values"] = np.stack([cache.values[i][0][:, :, :8] for i in (0, 13, 27)])
    out["generate"] = {
        "prompt": prompt, "sampler": CHAIN, "max_new_tokens": 10, "eos": eos,
        "uniforms": run["uniforms"], "tokens": run["tokens"],
        "text": tokenizer.decode(run["tokens"]),
    }  # fmt: skip
    print(f"generate: {out['generate']['text']!r}")

    # A streamed, greedy reply to the second chat, cut by a stop string from its text.
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Count from one to ten in words."},
    ]
    prompt = tokenizer.encode(template_text(reference, messages))
    run = sample_run(lm, prompt, 12, GREEDY, 22, eos)
    full = "".join(pieces_of(tokenizer, run["tokens"], [])[0])
    print(f"stream: {full!r}")
    stops = ["\n\n", ", three"]  # the second one arrives in two tokens
    assert ", three" in full
    pieces, used = pieces_of(tokenizer, run["tokens"], stops)
    out["stream"] = {
        "messages": messages, "prompt_tokens": len(prompt), "sampler": GREEDY,
        "max_new_tokens": 12, "eos": eos, "uniforms": run["uniforms"], "stop": stops,
        "tokens": run["tokens"], "used": used, "pieces": pieces, "full_text": full,
    }  # fmt: skip
    print(f"stream with stop {stops[0]!r}: used {used}, {pieces!r}")

    np.savez_compressed(HERE / "granite-4.0-350m.npz", **arrays)
    (HERE / "granite-4.0-350m.json").write_text(
        json.dumps(out, ensure_ascii=True, indent=1) + "\n", "utf-8"
    )
    for name in ("granite-4.0-350m.npz", "granite-4.0-350m.json"):
        print(f"wrote {name} ({(HERE / name).stat().st_size} bytes)")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    asset = assets.resolve(SMALL, assets.models_dir(find_root()))
    granite_tokenizer = AutoTokenizer.from_pretrained(asset)
    tiny(granite_tokenizer)
    granite(asset, granite_tokenizer)
