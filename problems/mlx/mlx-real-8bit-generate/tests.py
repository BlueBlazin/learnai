"""Tests for mlx-real-8bit-generate.

Every expected logit and token comes from `mlx-lm` with its floating parameters in
float32, stored by `fixtures/make.py`; the reference solution is never the oracle.

- The cases marked `assets=[]` use two three-layer quantized Granite models with a
  256-token vocabulary: `a` with the constants of Granite 4.2 3B (multipliers 1, 8
  bits in groups of 64), `b` with other multipliers, RoPE base 10,000, and 4 bits in
  groups of 32. Their checkpoints are drawn here from the same seeds as in `make.py`
  (random codes, scales, and biases) and written to a temporary directory as a
  `config.json` and a `model.safetensors` in the layout `mlx_lm.convert -q` writes.
- The other cases load the real 8-bit checkpoint of Granite 4.2 3B, 3.9 GB.

What is checked:

1. `load_model`: the parameter names are the tensor names of the checkpoint; packed
   codes are still uint32 with their stored shape; every floating parameter is
   float32; everything is evaluated.
2. the forward pass: logits against the fixture; the lazy graph of one call (one
   `QuantizedMatmul` per linear layer, no `Matmul`, at most one dequantization, which
   is the embedding lookup of the given tokens); a cached continuation against a pass
   over the whole sequence.
3. `generate`: the tokens, exactly; the calls it made to the model (a counting wrapper:
   the prompt once, then one token per call); and where it evaluated (`EvalSpy`: no
   `mx.eval`, one read per token, step `t + 1` scheduled before token `t` is read).
4. on the real model: peak memory of loading and generating, on the Apple GPU.
"""

import io
import json
import re
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from learnai.judge import case, check
from learnai.kit import EvalSpy, Guarded, check_guard, guard, mlx_memory
from mlx import nn

ASSET = "granite-4.2-3b-mlx-8bit"
REVISION = "7c630b526561db6b4641767c54595fe494176695"
TOL = {"rtol": 1e-3, "atol": 1e-3}  # as stated in Constraints
EOS = 100257  # <|im_end|> of the real model
PEAK_LIMIT_MIB = 6144  # 6 GiB
_cache: dict[object, object] = {}


def tiny_checkpoint(seed: int, config: dict) -> dict[str, np.ndarray]:
    """A random quantized Granite checkpoint in the MLX layout. Keep in step with make.py.

    Every linear layer and the embedding is stored as packed codes (uint32, lowest
    bits first) with one scale and one bias per group; the values are float32 numbers
    that bfloat16 represents exactly, so the file can store them as bfloat16.
    """
    rng = np.random.default_rng(seed)
    d, f, vocab = config["hidden_size"], config["intermediate_size"], config["vocab_size"]
    head = d // config["num_attention_heads"]
    kv = head * config["num_key_value_heads"]
    bits, group = config["quantization"]["bits"], config["quantization"]["group_size"]
    levels = 1 << bits
    code_std = float(np.sqrt((levels * levels - 1) / 12.0))

    def bf16(values: np.ndarray) -> np.ndarray:
        raw = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        return (raw & np.uint32(0xFFFF0000)).view(np.float32)

    out: dict[str, np.ndarray] = {}

    def quantized(name: str, rows: int, cols: int, std: float) -> None:
        codes = rng.integers(0, levels, (rows, cols)).astype(np.uint64)
        per_word = 32 // bits
        shifts = np.arange(per_word, dtype=np.uint64) * np.uint64(bits)
        words = (codes.reshape(rows, cols // per_word, per_word) << shifts).sum(axis=-1)
        scales = std / code_std * rng.uniform(0.5, 1.5, (rows, cols // group))
        scales = bf16(scales * rng.choice([-1.0, 1.0], scales.shape))
        biases = bf16(-scales * (levels - 1) * rng.uniform(0.35, 0.65, scales.shape))
        out[name + ".weight"] = words.astype(np.uint32)
        out[name + ".scales"] = scales
        out[name + ".biases"] = biases

    def gain(name: str) -> None:
        out[name + ".weight"] = bf16(rng.uniform(0.5, 1.5, d))

    quantized("model.embed_tokens", vocab, d, 1.0)
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        gain(p + "input_layernorm")
        quantized(p + "self_attn.q_proj", d, d, 0.5)
        quantized(p + "self_attn.k_proj", kv, d, 0.5)
        quantized(p + "self_attn.v_proj", kv, d, 0.1)
        quantized(p + "self_attn.o_proj", d, d, 0.1)
        gain(p + "post_attention_layernorm")
        quantized(p + "mlp.gate_proj", f, d, 0.1)
        quantized(p + "mlp.up_proj", f, d, 0.1)
        quantized(p + "mlp.down_proj", d, f, 0.1)
    gain("model.norm")
    if not config["tie_word_embeddings"]:
        quantized("lm_head", vocab, d, 0.3)
    return out


def graph_ops(array):
    """Names of the operations still pending behind `array` (empty: evaluated)."""
    dot = io.StringIO()
    mx.export_to_dot(dot, array)
    return re.findall(r'label ="(\w+)"', dot.getvalue())


def scratch(ctx):
    """A temporary directory for the running case, removed when the case ends."""
    key = ("dir", id(ctx))
    if key not in _cache:
        path = Path(tempfile.mkdtemp(prefix="learnai-granite8-"))
        _cache[key] = path

        def remove():
            shutil.rmtree(path, ignore_errors=True)
            _cache.pop(key, None)

        ctx.defer(remove)
    return _cache[key]


def tiny(ctx, name):
    """(directory, config, fixture) of the small model `a` or `b`: a checkpoint
    directory written for the running case."""
    key = f"tiny_{name}"
    if key not in _cache:
        spec = ctx.fixture("tiny-configs.json")[name]
        tensors = tiny_checkpoint(spec["seed"], spec["config"])
        total = float(sum(np.abs(t.astype(np.float64)).sum() for t in tensors.values()))
        fixture = ctx.fixture("tiny.npz")
        check.that(
            abs(total - float(fixture[f"checksum_{name}"])) < 1e-9 * total,
            "test setup: the random checkpoint differs from the one the fixture was made "
            "with (numpy changed its generator?); rerun fixtures/make.py",
        )
        stored = {
            n: mx.array(t) if t.dtype == np.uint32 else mx.array(t).astype(mx.bfloat16)
            for n, t in tensors.items()
        }
        mx.eval(stored)
        _cache[key] = (stored, spec["config"], fixture)
    stored, config, fixture = _cache[key]
    directory = scratch(ctx) / name
    if not directory.exists():
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config, indent=2), "utf-8")
        mx.save_safetensors(
            str(directory / "model.safetensors"), stored, metadata={"format": "mlx"}
        )
    return directory, config, fixture


def check_tree(model, directory):
    """The parameters are the checkpoint's tensors: packed codes kept, floats in float32."""
    check.that(isinstance(model, nn.Module), "load_model: must return an mlx.nn.Module")
    stored = mx.load(str(Path(directory) / "model.safetensors"))  # lazy: names and shapes
    have = dict(tree_flatten(model.parameters()))
    missing = sorted(set(stored) - set(have))
    extra = sorted(set(have) - set(stored))
    check.that(
        not missing and not extra,
        "load_model: the parameter names of the model are not the tensor names of the "
        f"checkpoint ({len(missing)} tensors have no parameter, {len(extra)} parameters "
        "have no tensor)",
        f"tensors without a parameter, first 5 of {len(missing)}: {missing[:5]}\n"
        f"parameters without a tensor, first 5 of {len(extra)}: {extra[:5]}",
    )
    for name, tensor in stored.items():
        value = have[name]
        packed = tensor.dtype == mx.uint32
        want = mx.uint32 if packed else mx.float32
        kind = "the packed codes must stay uint32" if packed else "floating parameters are float32"
        check.that(
            value.dtype == want and tuple(value.shape) == tuple(tensor.shape),
            f"load_model: parameter {name} has dtype {value.dtype} and shape "
            f"{tuple(value.shape)}, which is not what the checkpoint's tensor gives",
            f"expected {want} with shape {tuple(tensor.shape)}: {kind}",
        )
    pending = len(graph_ops(next(iter(have.values()))))
    check.that(pending == 0, "load_model: the parameters it returns are not evaluated")


def load(ctx, directory):
    model = ctx.solution.load_model(str(directory))
    check_tree(model, directory)
    return model


def logits_of(model, ids, cache=None):
    """(logits as float32 NumPy of shape (T, V), cache) for one call of the model."""
    tokens = mx.array([[int(t) for t in ids]])
    result = model(tokens) if cache is None else model(tokens, cache)
    check.that(
        isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], mx.array),
        "the model must return a pair (logits, cache) with logits an mx.array",
    )
    logits, cache = result
    check.that(
        logits.dtype == mx.float32,
        f"logits: got dtype {logits.dtype}, which is not the expected dtype",
        "expected float32: load_model converts the floating parameters",
    )
    got = np.array(logits)
    check.that(
        got.ndim == 3 and got.shape[:2] == (1, len(ids)),
        f"logits: got shape {got.shape}, which is not (1, T, vocab_size) for the T tokens passed",
    )
    return got[0], cache


class Counting(Guarded):
    """The model as `generate` sees it: the same calls, counted.

    Behind the kit's `Guarded`: the solution may call it; the model inside and the
    record of the calls are refused.
    """

    label = "model"
    refused = Guarded.refused | {"model", "shapes"}

    def __init__(self, model):
        self.model = model
        self.shapes = []

    def __call__(self, tokens, cache=None):
        self.shapes.append(tuple(tokens.shape) if isinstance(tokens, mx.array) else None)
        return self.model(tokens) if cache is None else self.model(tokens, cache)


def first_order_violation(kinds, n):
    """Index of the first token that was read before the step after it was scheduled."""
    scheduled = reads = 0
    for kind in kinds:
        if kind == "async_eval":
            scheduled += 1
        elif kind == "read":
            if scheduled < min(reads + 2, n):
                return reads
            reads += 1
    return None


def run_generate(ctx, model, prompt, n, eos_id, want, label="generate"):
    """generate(...) on a counting wrapper; check tokens, model calls, and evaluations."""
    prompt, want = [int(t) for t in prompt], [int(t) for t in want]
    counted = Counting(model)
    guard(ctx)
    with EvalSpy() as spy:
        got = ctx.solution.generate(counted, list(prompt), n, eos_id)
    check_guard()
    mx.synchronize()  # a step scheduled past an EOS may still be running
    check.that(
        isinstance(got, list) and all(type(t) is int for t in got),
        f"{label}: must return a list of Python ints",
        f"got {got!r}"[:300],
    )
    check.equal(got, want, label=f"{label}: tokens")

    # One call for the prompt, one per later token. Stopping at an EOS before the limit
    # leaves behind the one step that was scheduled before the EOS was read.
    limit = len(want) + (1 if want[-1] == eos_id and len(want) < n else 0)
    check.that(
        counted.shapes[:1] == [(1, len(prompt))]
        and all(shape == (1, 1) for shape in counted.shapes[1:]),
        f"{label}: the model must be called once with the whole prompt, shape (1, P), and "
        "then with one token per call, shape (1, 1), and the cache",
        f"token shapes of the calls: {counted.shapes}",
    )
    check.at_most("forward_calls", len(counted.shapes), limit)
    kinds = spy.kinds()
    check.that(
        spy.evals == 0,
        f"{label}: called mx.eval {spy.evals} time(s); it blocks until the step is computed. "
        "Schedule with mx.async_eval",
        spy.describe(),
    )
    late = first_order_violation(kinds, n)
    check.that(
        late is None,
        f"{label}: token {late} was read before the step after it was scheduled with "
        "mx.async_eval, so the GPU is idle while Python handles the token",
        spy.describe(),
    )
    check.at_most("reads", spy.implicit, len(want))
    check.at_most("async_evals", spy.async_evals, limit)
    return got


# ---------------------------------------------------------------------------------
# Two three-layer quantized models: no download needed.
# ---------------------------------------------------------------------------------

# The worked example of the statement: model a, prompt 0.
WORKED_LOGITS = [2.323889, 3.797198, -0.293408, 0.827526]
WORKED_TOKENS = [219, 154, 179, 99, 254, 82, 65, 155, 253, 192, 99, 209]


@case("the worked example: a three-layer 8-bit model, logits and 12 generated tokens",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    directory, _, fixture = tiny(ctx, "a")
    model = load(ctx, directory)
    prompt = fixture["prompt_0"]
    got, _ = logits_of(model, prompt)
    check.allclose(got, fixture["logits_a"], **TOL, label="logits of the prompt")
    check.allclose(got[-1, :4], WORKED_LOGITS, **TOL, label="logits[0, -1, :4]")
    tokens = run_generate(ctx, model, prompt, 12, None, fixture["tokens_a_0"])
    check.equal(tokens, WORKED_TOKENS, label="generate: tokens")


# Rejects dequantizes_at_load (in check_tree) and dequantizes_every_call on any machine.
@case("the weights stay packed: one QuantizedMatmul per linear layer and no float matrix",
      public=True, assets=[])  # fmt: skip
def stays_packed(ctx):
    for name in ("a", "b"):
        directory, config, fixture = tiny(ctx, name)
        model = load(ctx, directory)
        tokens = mx.array([fixture["prompt_0"].tolist()])
        mx.eval(tokens)
        logits, _ = model(tokens)
        ops = graph_ops(logits)
        n = config["num_hidden_layers"]
        detail = f"operations in the graph of one forward pass: {sorted(set(ops))}"
        check.that(
            ops.count("QuantizedMatmul") == 7 * n + 1,
            f"model {name}: the graph of a forward pass holds {ops.count('QuantizedMatmul')} "
            "QuantizedMatmul operations, which is not one per linear layer",
            f"expected {7 * n + 1}: seven per layer and the head\n{detail}",
        )
        # MLX names the operation behind mx.dequantize "Quantize" in an exported graph.
        dequantized = ops.count("Quantize") + ops.count("Dequantize")
        check.that(
            "Matmul" not in ops and dequantized <= 1,
            f"model {name}: the forward pass multiplies by a float matrix (Matmul: "
            f"{ops.count('Matmul')}, dequantizations: {dequantized}); only the embedding rows "
            "of the given tokens may be dequantized",
            detail,
        )
        check.that(
            ops.count("ScaledDotProductAttention") == n,
            f"model {name}: the graph holds {ops.count('ScaledDotProductAttention')} "
            "ScaledDotProductAttention operations, which is not one per layer",
            detail,
        )


# Rejects assumes_8_bits_groups_of_64 and multipliers_assumed_one.
@case("another configuration: 4 bits in groups of 32, other multipliers, RoPE base 10,000",
      public=True, assets=[])  # fmt: skip
def other_configuration(ctx):
    directory, _, fixture = tiny(ctx, "b")
    model = load(ctx, directory)
    got, _ = logits_of(model, fixture["prompt_0"])
    check.allclose(got, fixture["logits_b"], **TOL, label="logits of the prompt")
    run_generate(ctx, model, fixture["prompt_0"], 12, None, fixture["tokens_b_0"])


# Rejects rope_offset_missing without generating.
@case("the cache: a prefix, then the rest with the cache, equals one pass over everything",
      public=True, assets=[])  # fmt: skip
def cache_consistency(ctx):
    directory, _, fixture = tiny(ctx, "a")
    model = load(ctx, directory)
    prompt = fixture["prompt_0"].tolist()
    want = fixture["logits_a"]
    _, cache = logits_of(model, prompt[:3])
    rest, cache = logits_of(model, prompt[3:], cache)  # three tokens at once
    check.allclose(rest, want[3:], **TOL, label="logits of tokens 3 to 5 after a cached prefix")
    _, cache = logits_of(model, prompt[:5])
    last, _ = logits_of(model, prompt[5:], cache)  # a single token
    check.allclose(last, want[5:], **TOL, label="logits of the last token after a cached prefix")


# Rejects eos_ignored; the limit on calls allows the one step scheduled before the EOS.
@case("generation stops after the EOS token: 5 tokens of a limit of 12", public=True,
      assets=[])  # fmt: skip
def stops_at_eos(ctx):
    directory, _, fixture = tiny(ctx, "a")
    model = load(ctx, directory)
    want = fixture["tokens_a_1"].tolist()
    if want[4] != 2 or 2 in want[:4]:
        raise RuntimeError("authoring error: prompt 1 of model a no longer generates token 2 fifth")
    run_generate(ctx, model, fixture["prompt_1"], 12, 2, want[:5])


# Hidden cases. Their labels say nothing about the inputs: a hidden case's message is shown.


# Rejects no_kv_cache, sync_eval_then_read, and extra_step_at_limit by their counts.
@case("other prompts: 2 and 11 tokens long, both models", assets=[])
def other_prompts(ctx):
    for name in ("a", "b"):
        directory, _, fixture = tiny(ctx, name)
        model = load(ctx, directory)
        for index in (1, 2):
            run_generate(ctx, model, fixture[f"prompt_{index}"], 12, None,
                         fixture[f"tokens_{name}_{index}"])  # fmt: skip


@case("max_new_tokens = 1, an EOS generated exactly at the limit, an EOS only in the prompt",
      assets=[])  # fmt: skip
def limits(ctx):
    directory, _, fixture = tiny(ctx, "a")
    model = load(ctx, directory)
    want = fixture["tokens_a_0"].tolist()
    prompt = fixture["prompt_0"]
    run_generate(ctx, model, prompt, 1, None, want[:1])
    # The fourth token is the EOS and the limit is 4: no step may follow it.
    run_generate(ctx, model, prompt, 4, want[3], want[:4])
    # Token 17 is in the prompt and is never generated: it stops nothing.
    run_generate(ctx, model, prompt, 6, 17, want[:6])
    run_generate(ctx, model, prompt, 6, want[0], want[:1])


@case("a cached continuation in the 4-bit model, and two models loaded side by side",
      assets=[])  # fmt: skip
def two_models(ctx):
    dir_a, _, fixture = tiny(ctx, "a")
    dir_b, _, _ = tiny(ctx, "b")
    model_b = load(ctx, dir_b)
    model_a = load(ctx, dir_a)
    prompt = fixture["prompt_0"].tolist()
    _, cache = logits_of(model_b, prompt[:2])
    rest, _ = logits_of(model_b, prompt[2:], cache)
    check.allclose(rest, fixture["logits_b"][2:], **TOL, label="logits")
    got, _ = logits_of(model_a, prompt)
    check.allclose(got, fixture["logits_a"], **TOL, label="logits")


# ---------------------------------------------------------------------------------
# granite-4.2-3b-mlx-8bit: 40 layers, 3.66 billion weights in 3.9 GB.
# ---------------------------------------------------------------------------------


def real_fixture(ctx):
    fixture = ctx.fixture("granite-4.2-3b-mlx-8bit.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    return fixture


def real_model(ctx):
    if "real" not in _cache:
        _cache["real"] = ctx.solution.load_model(str(ctx.asset(ASSET)))
    return _cache["real"]


def check_real(ctx, model, index, got, *, hidden):
    fixture = real_fixture(ctx)
    what = "the prompt" if hidden else repr(ctx.fixture("prompts.json")[index]["prompt"])
    logits, _ = logits_of(model, fixture[f"ids_{index}"])
    check.allclose(logits[-1, fixture["columns"]], fixture[f"logits_{index}"], **TOL,
                   label=f"next-token logits for {what}, at 64 vocabulary columns")  # fmt: skip
    top = fixture[f"top_ids_{index}"]
    check.allclose(logits[-1, top], fixture[f"top_logits_{index}"], **TOL,
                   label=f"the 8 largest next-token logits for {what}")  # fmt: skip
    check.equal(got, fixture[f"tokens_{index}"].tolist(), label=f"generated tokens for {what}")


# Rejects dequantizes_at_load on the real model: 13.6 GiB of float32 weights.
@case("Granite 4.2 3B: load and generate ' Tokyo. The capital of Brazil is Bras' within 6 GiB",
      public=True, requires="mlx-gpu", time_limit_s=120)  # fmt: skip
def real_capital(ctx):
    fixture = real_fixture(ctx)
    _cache.pop("real", None)
    mx.synchronize()
    with mlx_memory() as mem:
        model = ctx.solution.load_model(str(ctx.asset(ASSET)))
        got = ctx.solution.generate(model, fixture["ids_0"].tolist(), 8, EOS)
    check.equal(got, fixture["tokens_0"].tolist(), label="generated tokens")
    check.memory(mem, at_most_mib=PEAK_LIMIT_MIB)
    check_tree(model, ctx.asset(ASSET))
    check_real(ctx, model, 0, got, hidden=False)
    _cache["real"] = model


@case("Granite 4.2 3B: a chat prompt that ends in <think>", time_limit_s=120)
def real_chat(ctx):
    model = real_model(ctx)
    fixture = real_fixture(ctx)
    got = run_generate(ctx, model, fixture["ids_1"], 8, EOS, fixture["tokens_1"])
    check_real(ctx, model, 1, got, hidden=True)


@case("Granite 4.2 3B: a chat prompt answered in one word, then <|im_end|>", time_limit_s=120)
def real_eos(ctx):
    model = real_model(ctx)
    fixture = real_fixture(ctx)
    got = run_generate(ctx, model, fixture["ids_2"], 8, EOS, fixture["tokens_2"])
    check_real(ctx, model, 2, got, hidden=True)
    _cache.pop("real", None)  # release 3.9 GB
