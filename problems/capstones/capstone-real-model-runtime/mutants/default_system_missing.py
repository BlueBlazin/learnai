# BUG: a conversation without a system message is rendered without a system turn; the model's template always opens with one, using a default text.
import codecs
import json
import re
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import torch

from whetstone.kit import ops
from whetstone.kit.bpe import translate_pattern

# -- stage 1: weights from a memory-mapped safetensors file --------------------------------

FLOAT_DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}


def load_weights(file, names=None) -> dict[str, torch.Tensor]:
    # The header: 8 bytes of length, then that many bytes of JSON. Nothing else is read
    # until a tensor is asked for.
    n = int.from_bytes(file[0:8], "little")
    header = json.loads(bytes(file[8 : 8 + n]))
    header.pop("__metadata__", None)
    data_start = 8 + n  # every data_offsets pair counts from here, not from byte 0
    names = list(header) if names is None else list(names)
    for name in names:
        if name not in header:
            raise KeyError(name)  # before any tensor byte is read

    weights = {}
    for name in names:
        entry = header[name]
        begin, end = entry["data_offsets"]
        # A view of the mapped file: only these bytes are touched.
        stored = np.frombuffer(
            file[data_start + begin : data_start + end], FLOAT_DTYPES[entry["dtype"]]
        )
        if entry["dtype"] == "BF16":
            # bfloat16 is the upper half of a float32: shift the 16 bits back into place.
            values = (stored.astype(np.uint32) << 16).view(np.float32)
        else:
            values = stored.astype(np.float32)
        weights[name] = torch.from_numpy(values.reshape(entry["shape"]))
    return weights


# -- stage 2: the tokenizer and the chat template -------------------------------------------


def byte_alphabet() -> list[str]:
    """The character ByteLevel writes each of the 256 byte values as."""
    keep = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    chars = {byte: chr(byte) for byte in keep}
    spare = 0x100
    for byte in range(256):
        if byte not in chars:
            chars[byte] = chr(spare)
            spare += 1
    return [chars[byte] for byte in range(256)]


class Tokenizer:
    def __init__(self, tokenizer_json: dict):
        model = tokenizer_json["model"]
        self.vocab: dict[str, int] = model["vocab"]
        self.rank = {(left, right): i for i, (left, right) in enumerate(model["merges"])}
        added = tokenizer_json["added_tokens"]
        self.added = {token["content"]: token["id"] for token in added}
        self.special = {token["id"] for token in added if token["special"]}
        by_length = sorted(self.added, key=len, reverse=True)  # the longest match wins
        self.added_re = re.compile("|".join(map(re.escape, by_length))) if added else None
        pattern = tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.word_re = re.compile(translate_pattern(pattern))
        self.alphabet = byte_alphabet()
        byte_of = {char: byte for byte, char in enumerate(self.alphabet)}
        self.pieces = {i: bytes(byte_of[c] for c in token) for token, i in self.vocab.items()}
        self.pieces.update({i: content.encode("utf-8") for content, i in self.added.items()})

    def _bpe(self, word: str) -> list[int]:
        parts = list(word)
        while len(parts) > 1:
            best = min(pairwise(parts), key=lambda pair: self.rank.get(pair, len(self.rank)))
            if best not in self.rank:
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
        return [self.vocab[part] for part in parts]

    def _ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self.word_re.finditer(text):  # merges never cross a word
            word = "".join(self.alphabet[byte] for byte in match.group().encode("utf-8"))
            ids.extend(self._bpe(word))
        return ids

    def encode(self, text: str) -> list[int]:
        if self.added_re is None:
            return self._ordinary(text)
        ids: list[int] = []
        position = 0
        for match in self.added_re.finditer(text):  # added tokens are cut out first
            ids.extend(self._ordinary(text[position : match.start()]))
            ids.append(self.added[match.group()])
            position = match.end()
        ids.extend(self._ordinary(text[position:]))
        return ids

    def token_bytes(self, token: int) -> bytes:
        return self.pieces[token]

    def decode(self, ids) -> str:
        # Join the bytes first: one character may be spread over several tokens.
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")


DEFAULT_SYSTEM = (
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
)


def chat_prompt(messages: list[dict]) -> str:
    # A system turn always comes first; a leading system message supplies its text.
    system, rest = DEFAULT_SYSTEM, messages
    if messages[0]["role"] == "system":
        system, rest = messages[0]["content"] or DEFAULT_SYSTEM, messages[1:]
    turns = [(m["role"], m["content"]) for m in rest]
    if messages[0]["role"] == "system":
        turns.insert(0, ("system", system))
    text = "".join(
        f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
        for role, content in turns
    )
    return text + "<|start_of_role|>assistant<|end_of_role|>"  # the turn is left open


# -- stage 3: the forward pass --------------------------------------------------------------


def read_config(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=config["num_hidden_layers"],
        d_model=config["hidden_size"],
        n_heads=config["num_attention_heads"],
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["hidden_size"] // config["num_attention_heads"],
        d_ff=config["shared_intermediate_size"],
        attention_scale=config["attention_multiplier"],
        embedding_multiplier=config["embedding_multiplier"],
        residual_multiplier=config["residual_multiplier"],
        logits_scaling=config["logits_scaling"],
        norm_eps=config["rms_norm_eps"],
        rope_theta=config["rope_theta"],
        eos=config["eos_token_id"],
    )


def _layers(weights, cfg, tokens: torch.Tensor, cache: dict | None):
    """Run `tokens` (1-D) after the cached ones: the hidden states and the new cache."""
    seq = tokens.shape[0]
    past = 0 if cache is None else cache["length"]
    half = cfg.head_dim // 2
    group = cfg.n_heads // cfg.n_kv_heads
    # The new tokens continue where the cache ends: positions past .. past + seq - 1.
    positions = past + torch.arange(seq)
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq  # (T, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    visible = torch.arange(past + seq)[None, :] <= positions[:, None]  # (T, past + T)

    def rope(z):
        rotated = torch.cat([-z[..., half:], z[..., :half]], dim=-1)  # (a, b) -> (-b, a)
        return z * cos + rotated * sin

    def heads(z, n):
        return z.reshape(seq, n, cfg.head_dim).transpose(0, 1)  # (n, T, head_dim)

    x = weights["model.embed_tokens.weight"][tokens] * cfg.embedding_multiplier
    keys, values = [], []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        # A key is rotated once, for its own position, and cached rotated.
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        if cache is not None:
            # cat builds new tensors: the cache that was passed in stays as it was.
            k = torch.cat([cache["keys"][i][:, :past], k], dim=1)
            v = torch.cat([cache["values"][i][:, :past], v], dim=1)
        keys.append(k)  # at the width of the KV heads
        values.append(v)
        # Each KV head serves `group` consecutive query heads: expanded for use only.
        scores = q @ k.repeat_interleave(group, dim=0).transpose(-1, -2)
        scores = (scores * cfg.attention_scale).masked_fill(~visible, float("-inf"))
        mixed = ops.softmax(scores) @ v.repeat_interleave(group, dim=0)
        mixed = mixed.transpose(0, 1).reshape(seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate, then up
        hidden = torch.nn.functional.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    return x, {"length": past + seq, "keys": keys, "values": values}


def _head(weights, cfg, x):
    x = ops.rms_norm(x, weights["model.norm.weight"], cfg.norm_eps)
    return (x @ weights["model.embed_tokens.weight"].T) / cfg.logits_scaling  # the tied head


def forward(weights, config: dict, ids: torch.Tensor) -> torch.Tensor:
    cfg = read_config(config)
    x, _ = _layers(weights, cfg, ids, None)
    return _head(weights, cfg, x)


# -- stage 4: cached generation with the sampler chain --------------------------------------


def prefill(weights, config: dict, prompt_ids: torch.Tensor):
    cfg = read_config(config)
    x, cache = _layers(weights, cfg, prompt_ids, None)
    return _head(weights, cfg, x[-1]), cache  # only the last position chooses a token


def decode_step(weights, config: dict, token: int, cache: dict):
    cfg = read_config(config)
    x, cache = _layers(weights, cfg, torch.tensor([token]), cache)
    return _head(weights, cfg, x[-1]), cache


def sample_token(logits: torch.Tensor, sampler: dict, history: list[int], u: float) -> int:
    # 1. Penalties on the raw logits, for the tokens generated so far.
    occurrences = np.bincount(np.asarray(history, dtype=np.int64), minlength=logits.shape[-1])
    counts = torch.from_numpy(occurrences.astype(np.float32))
    seen = counts > 0
    penalty = sampler["repetition_penalty"]
    x = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)
    x = x - counts * sampler["frequency_penalty"] - seen * sampler["presence_penalty"]
    # 2-4. Truncation: each filter sees the distribution the previous one left.
    x = ops.top_k_filter(x, sampler["top_k"])
    if sampler["top_p"] < 1.0:
        x = ops.top_p_filter(x, sampler["top_p"])
    x = ops.min_p_filter(x, sampler["min_p"])
    # 5. Temperature last: it reshapes what survived, and has no say in what survives.
    probs = ops.softmax(x / sampler["temperature"])
    # 6. Inverse CDF: the first index whose cumulative sum exceeds u times the total.
    cdf = torch.cumsum(probs, dim=-1)
    return int((cdf <= u * cdf[-1]).sum())


def _tokens(weights, config, prompt_ids, max_new_tokens, sampler, uniforms):
    """Yield the generated tokens one by one; the model runs only when the next is asked for."""
    eos = config["eos_token_id"]
    logits, cache = prefill(weights, config, prompt_ids)  # the whole prompt, once
    history: list[int] = []
    for step in range(max_new_tokens):
        token = sample_token(logits, sampler, history, uniforms[step])
        if token == eos:
            return  # the end of the turn is not output and is not fed back
        history.append(token)
        yield token
        if step + 1 < max_new_tokens:  # the last token's logits would never be read
            logits, cache = decode_step(weights, config, token, cache)


def generate(weights, config: dict, prompt_ids, max_new_tokens: int, sampler: dict, uniforms):
    return list(_tokens(weights, config, prompt_ids, max_new_tokens, sampler, uniforms))


# -- stage 5: streaming text with stop sequences --------------------------------------------


def _visible(text: str, stops: list[str], final: bool) -> tuple[int, bool]:
    """How much of `text` may be shown, and whether a stop string ended the output."""
    best = None
    for stop in stops:
        start = text.find(stop)
        if start >= 0 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)  # the match that ends first; then the longest
    if best is not None:
        return best[1], True
    if final:
        return len(text), False
    held = 0
    for stop in stops:
        for size in range(min(len(stop) - 1, len(text)), held, -1):
            if text.endswith(stop[:size]):  # a proper prefix: it may still become a stop
                held = size
                break
    return len(text) - held, False


def stream_chat(
    weights, config: dict, tokenizer, messages, max_new_tokens: int, sampler: dict, uniforms, stop
):
    prompt_ids = torch.tensor(tokenizer.encode(chat_prompt(messages)))
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text, shown = "", 0
    for token in _tokens(weights, config, prompt_ids, max_new_tokens, sampler, uniforms):
        # Special tokens are fed to the model like any other but are not text.
        text += decoder.decode(b"" if token in tokenizer.special else tokenizer.token_bytes(token))
        end, stopped = _visible(text, stop, final=False)
        if end > shown:
            yield text[shown:end]
            shown = end
        if stopped:
            return  # leaving the loop closes the token generator: no further forward pass
    text += decoder.decode(b"", final=True)  # an unfinished character becomes U+FFFD
    end, _ = _visible(text, stop, final=True)
    if end > shown:
        yield text[shown:end]
