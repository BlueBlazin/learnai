"""Tests for capstone-real-model-runtime.

Every expected value comes from `fixtures/make.py`, which runs the kit's own
implementations (the numpy forward pass `RealLM`, the tokenizer `ByteLevelBPE`) and
cross-checks them against `transformers` and `tokenizers`. The reference solution is
never the oracle.

- The cases marked `assets=[]` run a complete two-layer model that is committed in
  `fixtures/tiny/`: a config with multipliers that differ from Granite's, a 382-token
  miniature of Granite's tokenizer, and an 85 kB checkpoint in bfloat16. Every stage
  can be solved against it without a download.
- The other cases run the real granite-4.0-350m from the asset directory.

The runtime is built from the solution's own parts: the weights that later stages
use are the ones the solution's `load_weights` returned, and stage 5 gets the
solution's `Tokenizer`. Both are loaded once per run and kept in `_state`. Solution
functions are called directly, with tensors of the solution's framework, because the
weights and the cache must not be converted back and forth.

The file that `load_weights` gets is a `CountingFile` behind the kit's `Guarded`: a
solution may slice it, take its length, and read from it, all counted. Its counters,
its path, and the mapping behind it are refused, and a second file cannot be opened.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from learnai.judge import case, check
from learnai.kit import (
    ByteLevelBPE,
    CountingFile,
    Guarded,
    SafetensorsFile,
    bf16_to_f32,
    check_guard,
    guard,
)

S1 = "1: weights from a safetensors file"
S2 = "2: tokenizer and chat template"
S3 = "3: forward pass"
S4 = "4: cached generation and sampling"
S5 = "5: streaming text and stop strings"

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
TOL = {"rtol": 1e-3, "atol": 1e-3}  # the tolerance stated in Constraints
_state: dict[str, object] = {}


# -- models ---------------------------------------------------------------------------------


class File(Guarded, CountingFile):
    """The file a solution gets: slices, `len`, and reads, all counted."""

    label = "file"
    refused = Guarded.refused | {"path"}

    def __repr__(self):  # CountingFile's shows the path, which would open the file again
        return "CountingFile(...)"


class Files:
    """One model directory: paths, the parsed JSON files, and the kit's view of the weights."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.checkpoint_path = self.directory / "model.safetensors"
        self.config = json.loads((self.directory / "config.json").read_text("utf-8"))
        self._tokenizer_json = None

    @property
    def tokenizer_json(self):
        if self._tokenizer_json is None:
            text = (self.directory / "tokenizer.json").read_text("utf-8")
            self._tokenizer_json = json.loads(text)
        return self._tokenizer_json

    def kit_checkpoint(self):
        return SafetensorsFile(self.checkpoint_path)


def files_of(ctx, which):
    key = f"files_{which}"
    if key not in _state:
        if which == "tiny":
            directory = ctx.fixture_path("tiny/config.json").parent
        else:
            directory = ctx.asset(ASSET)
        _state[key] = Files(directory)
    return _state[key]


def call_load(ctx, file, *args):
    guard(ctx)
    weights = ctx.solution.load_weights(file, *args)
    check_guard()
    check.that(
        isinstance(weights, dict),
        f"load_weights: returned a {type(weights).__name__}, not a dict",
    )
    return weights


def weights_of(ctx, which):
    """The solution's own weights of a model, loaded once per run."""
    key = f"weights_{which}"
    if key not in _state:
        file = File(files_of(ctx, which).checkpoint_path)
        _state[key] = call_load(ctx, file)
        _state[f"file_{which}"] = file  # keeps the mapping alive
    return _state[key]


def tokenizer_of(ctx, which):
    """The solution's own tokenizer of a model, built once per run."""
    key = f"tokenizer_{which}"
    if key not in _state:
        _state[key] = ctx.solution.Tokenizer(files_of(ctx, which).tokenizer_json)
    return _state[key]


def fixture_of(ctx, which):
    name = "tiny" if which == "tiny" else ASSET
    key = f"fixture_{which}"
    if key not in _state:
        arrays, cases = ctx.fixture(f"{name}.npz"), ctx.fixture(f"{name}.json")
        if which == "real":
            check.that(
                cases["revision"] == REVISION,
                "internal: the fixture was made from another revision of the model",
            )
        _state[key] = (arrays, cases)
    return _state[key]


def with_eos(config, eos):
    return config if eos == config["eos_token_id"] else {**config, "eos_token_id": eos}


def to_numpy(ctx, value, label):
    check.that(
        ctx.fw.is_tensor(value),
        f"{label}: got a {type(value).__name__}, not a tensor of the framework",
    )
    return np.asarray(ctx.fw.to_numpy(value))


# == Stage 1: weights from a safetensors file ================================================


def write_file(ctx, entries, order=None, metadata=None, pad=0):
    """Write a safetensors file to a temporary path. `entries`: name -> (dtype, array).

    `order` is the order of the tensors in the data section (default: header order),
    and `pad` puts that many unused bytes in front of the data of every tensor.
    """
    codes = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}
    blobs, offsets, position = [], {}, 0
    for name in order or list(entries):
        dtype, array = entries[name]
        data = np.ascontiguousarray(array, dtype=codes[dtype]).tobytes()
        position += pad
        blobs.append(b"\x7f" * pad + data)
        offsets[name] = [position, position + len(data)]
        position += len(data)
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dtype, array) in entries.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(np.shape(array)),
            "data_offsets": offsets[name],
        }
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    handle = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)  # noqa: SIM115
    handle.write(len(text).to_bytes(8, "little") + text + b"".join(blobs))
    handle.close()
    ctx.defer(lambda: Path(handle.name).unlink(missing_ok=True))
    sizes = {name: offsets[name][1] - offsets[name][0] for name in entries}
    return File(handle.name), 8 + len(text), sizes


def bf16(values):
    """bfloat16 patterns of values that bfloat16 represents exactly."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    assert not (bits & 0xFFFF).any(), "not exactly representable in bfloat16"
    return (bits >> 16).astype(np.uint16)


def check_tensor(ctx, weights, name, want, label):
    check.that(name in weights, f"{label}: the result has no tensor {name!r}")
    got = to_numpy(ctx, weights[name], f"{label}: {name}")
    check.that(
        got.dtype == np.float32,
        f"{label}: tensor {name!r} has dtype {got.dtype}; every tensor must be float32",
    )
    check.shape(got, np.shape(want), label=f"{label}: {name}")
    check.that(
        np.array_equal(got, np.asarray(want, dtype=np.float32)),
        f"{label}: the values of {name!r} are not the values stored in the file",
        f"expected (first entries): {np.ravel(want)[:8]}\n"
        f"     got (first entries): {np.ravel(got)[:8]}\n",
    )


def check_bytes(file, limit, label):
    check.cost(file.counters, "bytes_read", at_most=limit, unit="bytes")
    check.that(file.counters.snapshot()["bytes_read"] > 0, f"{label}: nothing was read from `file`")


EXAMPLE = {
    "w": ("F32", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
    "b": ("BF16", bf16([1.0, -2.5])),
    "h": ("F16", np.array([0.5, -0.25, 8.0], dtype=np.float16)),
}
EXAMPLE_VALUES = {
    "w": [[1.0, 2.0], [3.0, 4.0]],
    "b": [1.0, -2.5],
    "h": [0.5, -0.25, 8.0],
}


@case("the stage 1 example: three tensors, three stored types, all float32", public=True,
      assets=[], stage=S1)  # fmt: skip
def weights_example(ctx):
    # Rejects bf16_as_float16 and offsets_from_file_start.
    file, header_bytes, sizes = write_file(ctx, EXAMPLE, metadata={"format": "pt"})
    weights = call_load(ctx, file)
    check.equal(list(weights), ["w", "b", "h"], label="names, in header order")
    for name, want in EXAMPLE_VALUES.items():
        check_tensor(ctx, weights, name, want, "load_weights(file)")
    check_bytes(file, header_bytes + sum(sizes.values()), "load_weights(file)")


@case("the tiny checkpoint: 18 bfloat16 tensors, decoded exactly", public=True, assets=[],
      stage=S1)  # fmt: skip
def weights_tiny(ctx):
    files = files_of(ctx, "tiny")
    file = File(files.checkpoint_path)
    weights = call_load(ctx, file)
    kit = files.kit_checkpoint()
    check.equal(list(weights), kit.names(), label="names, in header order")
    for name in kit.names():
        want = bf16_to_f32(kit.raw(name)).reshape(kit.info(name)["shape"])
        check_tensor(ctx, weights, name, want, "the tiny checkpoint")
    check_bytes(file, len(file), "the tiny checkpoint")


@case("names=[...] reads the header and those tensors, and nothing else", public=True, assets=[],
      stage=S1)  # fmt: skip
def weights_subset(ctx):
    # Rejects reads_whole_file.
    files = files_of(ctx, "tiny")
    kit = files.kit_checkpoint()
    wanted = ["model.norm.weight", "model.layers.1.self_attn.k_proj.weight"]
    file = File(files.checkpoint_path)
    weights = call_load(ctx, file, wanted)
    check.equal(list(weights), wanted, label="names, in the order asked for")
    size = 0
    for name in wanted:
        info = kit.info(name)
        size += info["data_offsets"][1] - info["data_offsets"][0]
        want = bf16_to_f32(kit.raw(name)).reshape(info["shape"])
        check_tensor(ctx, weights, name, want, "load_weights(file, names)")
    check_bytes(file, kit.data_start + size, "load_weights(file, names)")

    file = File(files.checkpoint_path)
    guard(ctx)
    check.raises(KeyError, ctx.solution.load_weights, file, [wanted[0], "model.nope"],
                 label="load_weights with a name that is not in the file")  # fmt: skip
    check_guard()
    check.cost(file.counters, "bytes_read", at_most=kit.data_start, unit="bytes",
               metric="bytes_read_before_keyerror")  # fmt: skip
    file = File(files.checkpoint_path)
    guard(ctx)
    check.raises(KeyError, ctx.solution.load_weights, file, ["__metadata__"],
                 label="load_weights(file, ['__metadata__'])")  # fmt: skip
    check_guard()


@case("data in another order than the header, gaps, odd offsets, a scalar, an empty tensor",
      assets=[], stage=S1)  # fmt: skip
def weights_layout(ctx):
    rng = ctx.rng
    entries = {
        "scalar": ("BF16", bf16(3.5)),
        "first": ("F32", rng.standard_normal((3, 5)).astype(np.float32)),
        "empty": ("F32", np.zeros((0, 4), dtype=np.float32)),
        "half": ("F16", rng.standard_normal((2, 2, 3)).astype(np.float16)),
        "brain": ("BF16", bf16(rng.integers(-64, 64, (4, 7)) / 8.0)),
    }
    order = ["brain", "half", "scalar", "empty", "first"]
    for pad in (0, 3):
        file, header_bytes, sizes = write_file(ctx, entries, order=order, pad=pad)
        weights = call_load(ctx, file)
        label = f"a file with {pad} unused bytes before every tensor"
        check.equal(list(weights), list(entries), label=f"{label}: names, in header order")
        for name, (dtype, stored) in entries.items():
            want = bf16_to_f32(stored) if dtype == "BF16" else stored.astype(np.float32)
            check_tensor(ctx, weights, name, np.reshape(want, np.shape(stored)), label)
        check_bytes(file, header_bytes + sum(sizes.values()), label)
        file, header_bytes, sizes = write_file(ctx, entries, order=order, pad=pad)
        weights = call_load(ctx, file, ["half", "scalar"])
        check.equal(list(weights), ["half", "scalar"], label=f"{label}: names of a subset")
        check_bytes(file, header_bytes + sizes["half"] + sizes["scalar"], label)


@case("the real checkpoint: three tensors alone, then all 226", public=True, stage=S1)
def weights_real(ctx):
    files = files_of(ctx, "real")
    kit = files.kit_checkpoint()
    wanted = [
        "model.layers.27.self_attn.k_proj.weight",
        "model.norm.weight",
        "model.layers.0.shared_mlp.output_linear.weight",
    ]
    file = File(files.checkpoint_path)
    weights = call_load(ctx, file, wanted)
    size = 0
    for name in wanted:
        info = kit.info(name)
        size += info["data_offsets"][1] - info["data_offsets"][0]
        want = bf16_to_f32(kit.raw(name)).reshape(info["shape"])
        check_tensor(ctx, weights, name, want, "three tensors of the real checkpoint")
    check_bytes(file, kit.data_start + size, "three tensors of the real checkpoint")

    # The whole model. Later stages use these weights.
    weights = weights_of(ctx, "real")
    check.equal(len(weights), 226, label="number of tensors")
    check.equal(list(weights), kit.names(), label="names, in header order")
    for name in ("model.embed_tokens.weight", "model.layers.13.self_attn.q_proj.weight"):
        rows = slice(0, 64)
        got = to_numpy(ctx, weights[name][rows], f"the real checkpoint: {name}")
        check.shape(to_numpy(ctx, weights[name], name), kit.info(name)["shape"], label=name)
        want = bf16_to_f32(kit.raw(name)[rows])
        check.that(
            got.dtype == np.float32 and np.array_equal(got, want),
            f"the real checkpoint: the first rows of {name!r} are not the stored values",
        )
    file = _state["file_real"]
    check.cost(file.counters, "bytes_read", at_most=len(file), unit="bytes",
               metric="bytes_read_whole_model")  # fmt: skip


# == Stage 2: tokenizer and chat template ====================================================


def check_ids(got, want, label, detail=""):
    check.that(
        isinstance(got, list) and all(type(t) is int for t in got),
        f"{label}: must return a list of Python ints",
        f"got: {got!r}"[:300],
    )
    check.equal(got, want, label=label)


def check_texts(ctx, which, entries):
    tokenizer = tokenizer_of(ctx, which)
    for entry in entries:
        text = entry["text"]
        check_ids(tokenizer.encode(text), entry["ids"], f"encode({text!r})")
        got = tokenizer.decode(entry["ids"])
        check.that(
            got == entry["decoded"],
            f"decode(encode({text!r})): got {got!r}, which is not the text of these tokens",
        )


def check_chats(ctx, which, entries):
    tokenizer = tokenizer_of(ctx, which)
    for entry in entries:
        messages = entry["messages"]
        got = ctx.solution.chat_prompt(messages)
        check.that(isinstance(got, str), "chat_prompt: must return a str")
        check.that(
            got == entry["text"],
            f"chat_prompt: wrong prompt text for a conversation of {len(messages)} messages",
            f"messages: {messages!r}\nexpected: {entry['text']!r}\n     got: {got!r}\n",
        )
        check_ids(
            tokenizer.encode(got),
            entry["ids"],
            f"encode(chat_prompt(...)), {len(messages)} messages",
        )


@case("the stage 2 example: ids, text, and bytes with the miniature tokenizer", public=True,
      assets=[], stage=S2)  # fmt: skip
def tokenizer_example(ctx):
    # Rejects ignores_pretokenizer.
    _, cases = fixture_of(ctx, "tiny")
    tokenizer = tokenizer_of(ctx, "tiny")
    text = "Hello there! How can I help you today?"
    want = [39, 283, 288, 267, 261, 0, 273, 296, 265, 256, 374, 322, 220, 373, 260, 292, 339, 30]
    assert cases["texts"][0] == {"text": text, "ids": want, "decoded": text}
    check_ids(tokenizer.encode(text), want, f"encode({text!r})")
    check.equal(tokenizer.decode(want), text, label="decode of those ids")
    for token, piece in [(283, b"el"), (267, b" the"), (220, b" "), (379, b"<|end_of_text|>"),
                         (172, b"\xf0")]:  # fmt: skip
        got = tokenizer.token_bytes(token)
        check.that(
            isinstance(got, bytes) and got == piece,
            f"token_bytes({token}): got {got!r}, which is not the bytes of this token",
        )
    check_texts(ctx, "tiny", cases["texts"][:3])


@case("the chat template: a system turn always, and an open assistant turn", public=True,
      assets=[], stage=S2)  # fmt: skip
def chat_template(ctx):
    # Rejects chatml_template and default_system_missing.
    _, cases = fixture_of(ctx, "tiny")
    check_chats(ctx, "tiny", cases["chats"])


@case("added tokens, characters of several bytes, and bytes that are not UTF-8", assets=[],
      stage=S2)  # fmt: skip
def tokenizer_edges(ctx):
    _, cases = fixture_of(ctx, "tiny")
    check_texts(ctx, "tiny", cases["texts"])
    tokenizer = tokenizer_of(ctx, "tiny")
    # The four bytes of an emoji, one token each; cut anywhere, the rest is U+FFFD.
    emoji = [172, 253, 247, 224]
    for ids, want in [(emoji, "\U0001f642"), (emoji[:2], "\ufffd"), ([39, *emoji[1:]], "H\ufffd\ufffd\ufffd"),
                      ([381, 379, 376], "<think><|end_of_text|><|pad|>"), ([], "")]:  # fmt: skip
        got = tokenizer.decode(ids)
        check.that(
            got == want, f"decode({ids}): got {got!r}, which is not the text of these tokens"
        )
    # A file with three more merges, which only words can stop: two spaces, and digits
    # past a group of three. Rejects ignores_pretokenizer. The ids are looked up.
    wider = json.loads(json.dumps(files_of(ctx, "tiny").tokenizer_json))
    for left, right, token in [("\u0120", "\u0120", 382), ("1", "2", 383), ("12", "3", 384),
                               ("123", "4", 385)]:  # fmt: skip
        wider["model"]["merges"].insert(0, [left, right])
        wider["model"]["vocab"][left + right] = token
    oracle = ByteLevelBPE.from_json(wider)
    third = ctx.solution.Tokenizer(wider)
    for text in ["a  b", "12345 1234", "   three spaces\n\n12 123"]:
        check_ids(third.encode(text), oracle.encode(text), f"encode({text!r}) with more merges")
    # A second tokenizer from a file with other ids must not disturb the first.
    other = json.loads(json.dumps(files_of(ctx, "tiny").tokenizer_json))
    other["added_tokens"][0]["content"] = "<|other|>"
    second = ctx.solution.Tokenizer(other)
    check_ids(second.encode("<|other|><|pad|>")[:1], [376], "encode with a second tokenizer")
    check_ids(tokenizer.encode("<|pad|>"), [376], "encode with the first tokenizer afterwards")


@case("the real tokenizer and the real prompts", public=True, stage=S2)
def tokenizer_real(ctx):
    _, cases = fixture_of(ctx, "real")
    check_texts(ctx, "real", cases["texts"])
    check_chats(ctx, "real", cases["chats"])
    tokenizer = tokenizer_of(ctx, "real")
    check.equal(tokenizer.token_bytes(100257), b"<|end_of_text|>", label="token_bytes(100257)")
    check.equal(tokenizer.token_bytes(220), b" ", label="token_bytes(220)")


# == Stage 3: forward pass ===================================================================


def run_forward(ctx, which, ids, label):
    files = files_of(ctx, which)
    out = ctx.solution.forward(weights_of(ctx, which), files.config, ctx.fw.tensor(ids))
    got = to_numpy(ctx, out, label)
    check.shape(got, (len(ids), files.config["vocab_size"]), label=f"{label}: logits")
    check.that(got.dtype == np.float32, f"{label}: the logits have dtype {got.dtype}")
    return got


def forward_tiny(ctx, index, label):
    arrays, _ = fixture_of(ctx, "tiny")
    ids = arrays[f"forward_ids_{index}"].astype(np.int64)
    got = run_forward(ctx, "tiny", ids, label)
    check.allclose(got, arrays[f"forward_logits_{index}"], **TOL, label=f"{label}: logits")


@case("the tiny model: logits at every position of an 18-token prompt", public=True, assets=[],
      stage=S3)  # fmt: skip
def forward_example(ctx):
    forward_tiny(ctx, 0, "forward")


@case("the tiny model: a prompt of one token", public=True, assets=[], stage=S3)
def forward_one_token(ctx):
    forward_tiny(ctx, 1, "forward, one token")


@case("the tiny model: one token repeated eight times", assets=[], stage=S3)
def forward_repeated(ctx):
    # Only positions (RoPE) and the causal mask distinguish the rows of this prompt.
    forward_tiny(ctx, 2, "forward, a repeated token")


@case("the real model: logits of a 12-token prompt", public=True, stage=S3)
def forward_real(ctx):
    arrays, _ = fixture_of(ctx, "real")
    ids = arrays["forward_ids"].astype(np.int64)
    got = run_forward(ctx, "real", ids, "forward on the real model")
    check.allclose(got[:, arrays["columns"]], arrays["forward_logits"], **TOL,
                   label="logits at 64 vocabulary columns")  # fmt: skip
    check.equal(
        got.argmax(-1), arrays["forward_argmax"], label="the argmax token of every position"
    )


# == Stage 4: cached generation and sampling =================================================


def check_cache(ctx, cache, config, length, keys, values, label, layers=None, width=None):
    """A cache of the statement's layout whose first `length` slots equal the fixture's."""
    check.that(
        isinstance(cache, dict) and {"length", "keys", "values"} <= set(cache),
        f'{label}: the cache must be a dict with "length", "keys", and "values"',
        f"got: {type(cache).__name__}",
    )
    check.that(
        type(cache["length"]) is int and cache["length"] == length,
        f"{label}: cache['length'] is {cache['length']!r}, which is not the number of tokens "
        "the cache holds",
    )
    n_layers = config["num_hidden_layers"]
    kv, head_dim = (
        config["num_key_value_heads"],
        config["hidden_size"] // config["num_attention_heads"],
    )
    for name, want_all in (("keys", keys), ("values", values)):
        tensors = cache[name]
        check.that(
            isinstance(tensors, (list, tuple)) and len(tensors) == n_layers,
            f"{label}: cache[{name!r}] must be a list with one tensor per layer",
        )
        for slot, layer in enumerate(range(n_layers) if layers is None else layers):
            got = to_numpy(ctx, tensors[layer], f"{label}: cache[{name!r}][{layer}]")
            check.that(
                got.ndim == 3
                and got.shape[0] == kv
                and got.shape[1] >= length
                and got.shape[2] == head_dim,
                f"{label}: cache[{name!r}][{layer}] has shape {got.shape}; it must be "
                "(key/value heads, capacity >= length, head_dim)",
                f"key/value heads: {kv}, length: {length}, head_dim: {head_dim}\n",
            )
            got = got[:, :length]
            want = want_all[slot]
            if width is not None:
                got = got[..., :width]
            check.allclose(got, want, **TOL, label=f"{label}: {name} of layer {layer}")


def logits_of(ctx, out, vocab, label):
    check.that(
        isinstance(out, tuple) and len(out) == 2,
        f"{label}: must return a tuple (logits, cache)",
        f"got: {type(out).__name__}",
    )
    got = to_numpy(ctx, out[0], f"{label}: logits")
    check.shape(got, (vocab,), label=f"{label}: logits")
    check.that(got.dtype == np.float32, f"{label}: the logits have dtype {got.dtype}")
    return got, out[1]


def fixture_cache(ctx, keys, values):
    """A cache built by the test, in the statement's layout, with capacity == length."""
    return {
        "length": int(keys.shape[2]),
        "keys": [ctx.fw.tensor(np.ascontiguousarray(k)) for k in keys],
        "values": [ctx.fw.tensor(np.ascontiguousarray(v)) for v in values],
    }


def run_generate(ctx, which, entry, label):
    files = files_of(ctx, which)
    got = ctx.solution.generate(
        weights_of(ctx, which),
        with_eos(files.config, entry["eos"]),
        ctx.fw.tensor(np.array(entry["prompt"], dtype=np.int64)),
        entry["max_new_tokens"],
        entry["sampler"],
        list(entry["uniforms"]),
    )
    check.that(
        isinstance(got, list) and all(type(t) is int for t in got),
        f"{label}: must return a list of Python ints",
        f"got: {got!r}"[:300],
    )
    check.equal(got, entry["tokens"], label=f"{label}: tokens")


def run_sampler(ctx, entries, label):
    for number, entry in enumerate(entries):
        logits = ctx.fw.tensor(np.array(entry["logits"], dtype=np.float32))
        got = ctx.solution.sample_token(
            logits, entry["sampler"], list(entry["history"]), entry["u"]
        )
        check.that(
            type(got) is int,
            f"{label}: sample_token must return a Python int, got a {type(got).__name__}",
        )
        check.that(
            got == entry["token"],
            f"{label}: row {number}: sample_token chose token {got}, which is not the token "
            "of the chain",
            f"logits = {entry['logits']}\nsampler = {entry['sampler']}\n"
            f"history = {entry['history']}\nu = {entry['u']}\nexpected: {entry['token']}\n",
        )


SAMPLER_EXAMPLE = {
    "repetition_penalty": 2.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.5,
    "top_k": 4,
    "top_p": 0.85,
    "min_p": 0.4,
    "temperature": 2.0,
}
EXAMPLE_LOGITS = [3.0, 2.0, 4.0, 0.0, 1.0, 2.5]


@case("the stage 4 example: one row of logits through the sampler chain", public=True, assets=[],
      stage=S4)  # fmt: skip
def sampler_example(ctx):
    # Rejects temperature_first.
    logits = ctx.fw.tensor(np.array(EXAMPLE_LOGITS, dtype=np.float32))
    for u, want in [(0.2, 0), (0.5, 0), (0.6, 5), (0.99, 5), (0.0, 0)]:
        got = ctx.solution.sample_token(logits, SAMPLER_EXAMPLE, [2, 2, 3], u)
        check.that(type(got) is int, "sample_token must return a Python int")
        check.equal(got, want, label=f"sample_token with u = {u}")
    _, cases = fixture_of(ctx, "tiny")
    run_sampler(ctx, cases["sampler"][:8], "rows of 24 logits")


@case("prefill: the logits of the last position and a cache at key/value width", public=True,
      assets=[], stage=S4)  # fmt: skip
def prefill_tiny(ctx):
    # Rejects cache_at_query_width.
    arrays, _ = fixture_of(ctx, "tiny")
    files = files_of(ctx, "tiny")
    prompt = arrays["cache_prompt"].astype(np.int64)
    out = ctx.solution.prefill(weights_of(ctx, "tiny"), files.config, ctx.fw.tensor(prompt))
    got, cache = logits_of(ctx, out, files.config["vocab_size"], "prefill")
    check.allclose(got, arrays["cache_prefill_logits"], **TOL, label="prefill: logits")
    check_cache(ctx, cache, files.config, len(prompt), arrays["cache_keys"],
                arrays["cache_values"], "prefill")  # fmt: skip


@case("decode_step continues a cache it did not build, one token at a time", public=True,
      assets=[], stage=S4)  # fmt: skip
def decode_steps_tiny(ctx):
    # Rejects new_token_position_zero: the test's cache holds 54 tokens, so the new
    # token is at position 54, and nothing but the cache says what came before.
    arrays, _ = fixture_of(ctx, "tiny")
    files = files_of(ctx, "tiny")
    weights = weights_of(ctx, "tiny")
    cache = fixture_cache(ctx, arrays["cache_keys"], arrays["cache_values"])
    tokens = [int(t) for t in arrays["cache_tokens"]]
    for i, token in enumerate(tokens):
        out = ctx.solution.decode_step(weights, files.config, token, cache)
        got, cache = logits_of(ctx, out, files.config["vocab_size"], f"decode_step {i + 1}")
        check.allclose(
            got, arrays["cache_step_logits"][i], **TOL, label=f"decode_step {i + 1}: logits"
        )
    check_cache(ctx, cache, files.config, arrays["cache_keys"].shape[2] + len(tokens),
                arrays["cache_final_keys"], arrays["cache_final_values"],
                f"the cache after {len(tokens)} steps")  # fmt: skip


@case("generate: greedy, then the full chain with supplied uniforms", public=True, assets=[],
      stage=S4)  # fmt: skip
def generate_tiny(ctx):
    _, cases = fixture_of(ctx, "tiny")
    by_name = {entry["name"]: entry for entry in cases["generate"]}
    run_generate(ctx, "tiny", by_name["greedy"], "generate, top_k = 1")
    run_generate(ctx, "tiny", by_name["chain"], "generate, every stage of the chain")


@case("the cache is a value: two continuations of one cache", assets=[], stage=S4)
def cache_branch(ctx):
    # Rejects cache_shared_buffer.
    arrays, _ = fixture_of(ctx, "tiny")
    files = files_of(ctx, "tiny")
    weights = weights_of(ctx, "tiny")
    vocab = files.config["vocab_size"]
    prompt = arrays["cache_prompt"].astype(np.int64)
    out = ctx.solution.prefill(weights, files.config, ctx.fw.tensor(prompt))
    _, start = logits_of(ctx, out, vocab, "prefill")
    first, other = int(arrays["cache_tokens"][0]), int(arrays["cache_other"])
    out = ctx.solution.decode_step(weights, files.config, first, start)
    got, after_first = logits_of(ctx, out, vocab, "decode_step(first token)")
    check.allclose(got, arrays["cache_step_logits"][0], **TOL, label="decode_step(first token)")
    out = ctx.solution.decode_step(weights, files.config, other, start)
    got, _ = logits_of(ctx, out, vocab, "decode_step(another token) from the same cache")
    check.allclose(got, arrays["cache_branch_logits"], **TOL,
                   label="logits of the second continuation of one cache")  # fmt: skip
    check_cache(ctx, start, files.config, len(prompt), arrays["cache_keys"],
                arrays["cache_values"], "the prefill cache after two decode steps from it")  # fmt: skip
    out = ctx.solution.decode_step(
        weights, files.config, int(arrays["cache_tokens"][1]), after_first
    )
    got, _ = logits_of(ctx, out, vocab, "the first continuation, continued")
    check.allclose(got, arrays["cache_step_logits"][1], **TOL,
                   label="logits of the first continuation after the second was computed")  # fmt: skip


@case("generate: other sampler settings, end-of-sequence, and a single token", assets=[],
      stage=S4)  # fmt: skip
def generate_more(ctx):
    # Rejects eos_returned and history_includes_prompt.
    _, cases = fixture_of(ctx, "tiny")
    for number, entry in enumerate(cases["generate"]):
        run_generate(ctx, "tiny", entry, f"generate, run {number}")
    run_sampler(ctx, cases["sampler"][8:], "more rows of 24 logits")


@case("the real model: prefill, cache, and ten sampled tokens", public=True, stage=S4)
def generate_real(ctx):
    arrays, cases = fixture_of(ctx, "real")
    files = files_of(ctx, "real")
    entry = cases["generate"]
    prompt = np.array(entry["prompt"], dtype=np.int64)
    out = ctx.solution.prefill(weights_of(ctx, "real"), files.config, ctx.fw.tensor(prompt))
    got, cache = logits_of(ctx, out, files.config["vocab_size"], "prefill on the real model")
    check.allclose(got[arrays["columns"]], arrays["cache_prefill_logits"], **TOL,
                   label="prefill: logits at 64 vocabulary columns")  # fmt: skip
    check_cache(ctx, cache, files.config, len(prompt), arrays["cache_keys"],
                arrays["cache_values"], "prefill on the real model",
                layers=[int(i) for i in arrays["cache_layers"]], width=8)  # fmt: skip
    run_generate(ctx, "real", entry, "generate on the real model")


# == Stage 5: streaming text and stop strings ================================================


def run_stream(ctx, which, entry, label):
    files = files_of(ctx, which)
    stream = ctx.solution.stream_chat(
        weights_of(ctx, which),
        with_eos(files.config, entry["eos"]),
        tokenizer_of(ctx, which),
        entry["messages"],
        entry["max_new_tokens"],
        entry["sampler"],
        list(entry["uniforms"]),
        list(entry["stop"]),
    )
    check.that(
        hasattr(stream, "__next__"),
        f"{label}: stream_chat must be a generator (or return an iterator)",
        f"got: {type(stream).__name__}",
    )
    pieces = list(stream)
    check.that(
        all(isinstance(p, str) and p for p in pieces),
        f"{label}: every piece must be a non-empty str",
        f"got: {pieces!r}"[:400],
    )
    detail = (
        f"stop = {entry['stop']!r}, max_new_tokens = {entry['max_new_tokens']}\n"
        f"expected pieces: {entry['pieces']!r}\n     got pieces: {pieces!r}\n"
    )
    check.that(
        "".join(pieces) == "".join(entry["pieces"]),
        f"{label}: the streamed text is {''.join(pieces)!r}, which is not the visible reply",
        detail,
    )
    check.that(
        pieces == entry["pieces"],
        f"{label}: the text is right but it was cut into pieces differently",
        detail,
    )


def stream_cases(ctx, names=None):
    _, cases = fixture_of(ctx, "tiny")
    seen = {}
    for position, entry in enumerate(cases["stream"]):
        seen.setdefault(entry["name"], position)
    return [
        (f"reply {position}", entry)
        for position, entry in enumerate(cases["stream"])
        if names is None or (entry["name"] in names and seen[entry["name"]] == position)
    ]


@case("the stage 5 example: a reply cut at a stop string that spans tokens", public=True,
      assets=[], stage=S5)  # fmt: skip
def stream_example(ctx):
    # Rejects stop_within_piece and stop_prefix_shown.
    for label, entry in stream_cases(ctx, {"chat0_spanning", "chat0_decoy"}):
        run_stream(ctx, "tiny", entry, label)


@case("no stop string: unfinished characters, special tokens, the limit, end-of-sequence",
      public=True, assets=[], stage=S5)  # fmt: skip
def stream_plain(ctx):
    # Rejects decodes_tokens_alone.
    for label, entry in stream_cases(ctx, {"chat0_none", "chat0_limit", "chat0_eos"}):
        run_stream(ctx, "tiny", entry, label)


@case("stop strings that never complete are released, at the latest when the reply ends",
      assets=[], stage=S5)  # fmt: skip
def stream_released(ctx):
    # Rejects held_text_dropped.
    for label, entry in stream_cases(ctx, {"chat0_absent", "chat1_absent"}):
        run_stream(ctx, "tiny", entry, label)


@case("every recorded reply, with every stop list", assets=[], stage=S5)
def stream_all(ctx):
    for label, entry in stream_cases(ctx):
        run_stream(ctx, "tiny", entry, label)


@case("stop strings inside one token, at the first character, and overlapping", assets=[],
      stage=S5)  # fmt: skip
def stream_stop_edges(ctx):
    _, cases = fixture_of(ctx, "tiny")
    for position, entry in enumerate(cases["stream_edges"]):
        run_stream(ctx, "tiny", entry, f"edge case {position}")


@case("the real model: a streamed reply, cut at a stop string", public=True, stage=S5)
def stream_real(ctx):
    _, cases = fixture_of(ctx, "real")
    run_stream(ctx, "real", cases["stream"], "stream_chat on the real model")
