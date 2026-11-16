"""Tests for real-model-sizing.

The numbers for the two Granite checkpoints come from `fixtures/expected.json`, which
`fixtures/make.py` measured with `transformers`, `mlx-lm`, and the files themselves.
The 3B model's header and config are committed fixtures, so its cases need no
download; the 350M model's header is read from the downloaded checkpoint. Synthetic
headers are built here and their expected numbers are worked out by hand in the
comments; only the random sweep uses the reference.
"""

import copy
import json
from math import prod

from learnai.judge import case, check
from learnai.kit import SafetensorsFile

ASSET = "granite-4.0-350m"
LARGE = "granite-4.2-3b-mlx-8bit"
KEYS = ("parameters", "tensor_bytes", "kv_bytes_per_token", "quantized_bytes")
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I64": 8}


def table(tensors):
    """A safetensors tensor table for (name, dtype, shape) triples, stored back to back."""
    header, offset = {}, 0
    for name, dtype, shape in tensors:
        size = prod(shape) * ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}  # fmt: skip
        offset += size
    return header


def decoder(vocab, d, layers, heads, kv_heads, ff, *, head=False, dtype="BF16"):
    """The tensors of a Llama-style decoder with head_dim = d // heads."""
    hd = d // heads
    tensors = [("model.embed_tokens.weight", dtype, (vocab, d))]
    for i in range(layers):
        p = f"model.layers.{i}."
        tensors += [
            (p + "input_layernorm.weight", dtype, (d,)),
            (p + "self_attn.q_proj.weight", dtype, (heads * hd, d)),
            (p + "self_attn.k_proj.weight", dtype, (kv_heads * hd, d)),
            (p + "self_attn.v_proj.weight", dtype, (kv_heads * hd, d)),
            (p + "self_attn.o_proj.weight", dtype, (d, heads * hd)),
            (p + "post_attention_layernorm.weight", dtype, (d,)),
            (p + "mlp.gate_proj.weight", dtype, (ff, d)),
            (p + "mlp.up_proj.weight", dtype, (ff, d)),
            (p + "mlp.down_proj.weight", dtype, (d, ff)),
        ]
    tensors.append(("model.norm.weight", dtype, (d,)))
    if head:
        tensors.append(("lm_head.weight", dtype, (vocab, d)))
    return tensors


def packed(tensors, bits, group):
    """The same tensors as an MLX quantizer stores them: matrices become packed uint32
    codes plus bfloat16 scales and biases; vectors are kept."""
    out = []
    for name, dtype, shape in tensors:
        if len(shape) != 2:
            out.append((name, dtype, shape))
            continue
        base = name.removesuffix(".weight")
        rows, cols = shape
        out += [
            (base + ".scales", "BF16", (rows, cols // group)),
            (base + ".weight", "U32", (rows, cols * bits // 32)),
            (base + ".biases", "BF16", (rows, cols // group)),
        ]
    return out


def config(d, layers, heads, kv_heads, tied, quantization=None):
    cfg = {
        "model_type": "granite",
        "hidden_size": d,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "intermediate_size": 128,
        "tie_word_embeddings": tied,
        "vocab_size": 100,
    }
    if quantization is not None:
        cfg["quantization"] = {"group_size": quantization[1], "bits": quantization[0], "mode": "affine"}  # fmt: skip
    return cfg


def report(ctx, header, cfg, label):
    """Call the solution on copies and check the shape of the answer before its numbers."""
    header_before, cfg_before = copy.deepcopy(header), copy.deepcopy(cfg)
    got = ctx.solution.size_report(header, cfg)
    check.that(isinstance(got, dict), f"{label}: size_report must return a dict")
    check.that(
        set(got) == set(KEYS),
        f"{label}: the result must have exactly the keys {', '.join(KEYS)}",
        f"got keys: {sorted(got, key=str)!r}",
    )
    check.that(
        all(type(got[key]) is int for key in KEYS),
        f"{label}: every value must be a Python int",
        f"got: {got!r}",
    )
    check.that(
        header == header_before and cfg == cfg_before,
        f"{label}: size_report modified its arguments",
    )
    return got


def check_literal(ctx, header, cfg, expected, label):
    got = report(ctx, header, cfg, label)
    check.equal(got, dict(zip(KEYS, expected, strict=True)), label=label)


# The tiny model of the statement: vocabulary 100, width 64, one block, 2 query heads on
# 1 KV head of width 32, SwiGLU 128.
#   matrices: 6400 + 4096 + 2048 + 2048 + 4096 + 3 * 8192 = 43264; three norms: 192.
#   8-bit: 43264 * 17 / 16 + 192 * 2 = 45968 + 384 = 46352.
TINY = {"vocab": 100, "d": 64, "layers": 1, "heads": 2, "kv_heads": 1, "ff": 128}


@case("the worked example: a one-block tied model in bfloat16", public=True, assets=[])
def worked_example(ctx):
    header = table(decoder(**TINY))
    cfg = config(64, 1, 2, 1, tied=True)
    check_literal(ctx, header, cfg, (43456, 86912, 128, 46352), "size_report(tiny)")


# Rejects stored_elements_as_parameters (1,029,450,240 stored elements), untied_head_dropped,
# kv_query_heads (40 heads), quant_overhead_ignored, scales_without_biases, norms_quantized,
# and bytes_from_parameters.
@case("Granite 4.2 3B as shipped at 8 bits: 3.66 B parameters in 927 tensors",
      public=True, assets=[])  # fmt: skip
def granite_3b_8bit(ctx):
    header = ctx.fixture(f"{LARGE}.header.json")
    cfg = ctx.fixture(f"{LARGE}.config.json")
    want = ctx.fixture("expected.json")[LARGE]
    check.that(len(header) == 927, "the fixture header is not the 8-bit 3B checkpoint's")
    check_literal(
        ctx, header, cfg, tuple(want[key] for key in KEYS), "size_report(granite-4.2-3b, 8-bit)"
    )


# Rejects tied_head_counted_twice (first call) and untied_head_dropped (second call).
@case("a file that stores the head next to the embedding: tied, then untied",
      public=True, assets=[])  # fmt: skip
def head_stored(ctx):
    header = table(decoder(**TINY, head=True))
    # Tied: lm_head is a second copy. Parameters and the prediction do not change; the
    # file is 100 * 64 * 2 = 12800 bytes larger.
    check_literal(
        ctx,
        header,
        config(64, 1, 2, 1, tied=True),
        (43456, 99712, 128, 46352),
        "size_report(tied model, matrix stored twice)",
    )
    # Untied: 6400 more parameters, and 6400 * 17 / 16 = 6800 more quantized bytes.
    check_literal(
        ctx,
        header,
        config(64, 1, 2, 1, tied=False),
        (49856, 99712, 128, 53152),
        "size_report(untied model)",
    )


# Rejects stored_elements_as_parameters on a file small enough to check by hand, and
# any solution that assumes 8 bits or four codes per word.
@case("the tiny model stored at 4 bits with groups of 32", public=True, assets=[])
def tiny_packed_4bit(ctx):
    plain = decoder(**TINY)
    header = table(packed(plain, bits=4, group=32))
    # On disk: codes 43264 / 2 bytes, scales and biases 2 * 2 * 43264 / 32, norms 384.
    check.that(sum(b - a for a, b in (t["data_offsets"] for t in header.values())) == 27424, "bug")
    cfg = config(64, 1, 2, 1, tied=True, quantization=(4, 32))
    check_literal(ctx, header, cfg, (43456, 27424, 128, 46352), "size_report(tiny, 4-bit)")
    # A tied 8-bit file that also stores a packed head: its three tensors are a copy.
    header = table(packed(decoder(**TINY, head=True), bits=8, group=64))
    cfg = config(64, 1, 2, 1, tied=True, quantization=(8, 64))
    check_literal(
        ctx, header, cfg, (43456, 46352 + 6800, 128, 46352), "size_report(tiny, 8-bit, head stored)"
    )


# Rejects bytes_from_parameters (float32 is 4 bytes) and norms_quantized.
@case("a float32 checkpoint, and tensors that are not matrices", assets=[])
def float32_and_other_ranks(ctx):
    tensors = decoder(vocab=200, d=128, layers=2, heads=4, kv_heads=2, ff=192, dtype="F32")
    tensors.append(("model.patch_embed.weight", "F32", (16, 3, 4, 4)))  # 768 elements, rank 4
    tensors.append(("model.scale", "F32", ()))  # a scalar
    header = table(tensors)
    # matrices: 25600 + 2 * (2 * 16384 + 2 * 8192 + 3 * 24576) = 271360; vectors: 5 * 128.
    parameters = 271360 + 640 + 768 + 1
    quantized = 271360 * 17 // 16 + 2 * (640 + 768 + 1)
    cfg = config(128, 2, 4, 2, tied=True)
    check_literal(
        ctx, header, cfg, (parameters, 4 * parameters, 512, quantized), "size_report(float32)"
    )
    # Without a "quantization" entry in the config, a uint32 tensor is just a tensor.
    odd = table([("model.vq_proj.weight", "U32", (8, 64)), ("model.vq_proj.scales", "BF16", (8,))])
    check_literal(
        ctx,
        odd,
        config(128, 2, 4, 2, tied=False),
        (520, 2064, 512, 512 + 32 + 16),
        "size_report(uint32 tensor, config without quantization)",
    )


# Rejects kv_query_heads and kv_keys_only on shapes where every factor differs.
@case("KV bytes for several layer counts, head counts, and head widths", assets=[])
def kv_shapes(ctx):
    header = table(decoder(**TINY))
    for d, layers, heads, kv_heads in ((64, 1, 2, 2), (192, 5, 6, 2), (2048, 28, 16, 8), (512, 3, 8, 1)):  # fmt: skip
        got = report(ctx, header, config(d, layers, heads, kv_heads, tied=True), "size_report")
        want = 2 * layers * kv_heads * (d // heads) * 2
        check.equal(
            got["kv_bytes_per_token"],
            want,
            label=f"kv_bytes_per_token with {layers} layers, {heads} heads, {kv_heads} KV heads, width {d}",
        )


@case("random models: plain and packed, tied and untied", assets=[])
def random_models(ctx):
    rng = ctx.rng
    for i in range(24):
        kv_heads = int(rng.integers(1, 4))
        heads = kv_heads * int(rng.integers(1, 4))
        d = 64 * heads
        model = decoder(
            vocab=int(rng.integers(50, 500)),
            d=d,
            layers=int(rng.integers(1, 5)),
            heads=heads,
            kv_heads=kv_heads,
            ff=64 * int(rng.integers(1, 6)),
            head=bool(rng.integers(2)),
            dtype=("BF16", "F32", "F16")[int(rng.integers(3))],
        )
        quantization = None
        if rng.integers(2):
            quantization = ((2, 4, 8)[int(rng.integers(3))], (32, 64)[int(rng.integers(2))])
            model = packed(model, *quantization)
        header = table(model)
        cfg = config(d, int(rng.integers(1, 40)), heads, kv_heads, bool(rng.integers(2)), quantization)  # fmt: skip
        got = report(ctx, header, cfg, f"size_report(random model {i})")
        check.equal(
            got, ctx.oracle.size_report(header, cfg), label=f"size_report(random model {i})"
        )


# ---------------------------------------------------------------------------------
# granite-4.0-350m: the header is read from the downloaded checkpoint.
# ---------------------------------------------------------------------------------


def small_model(ctx):
    want = ctx.fixture("expected.json")[ASSET]
    checkpoint = SafetensorsFile(ctx.asset(ASSET, "model.safetensors"))
    check.that(checkpoint.nbytes == want["file_bytes"], "the checkpoint on disk has another size")
    cfg = json.loads(ctx.asset(ASSET, "config.json").read_text(encoding="utf-8"))
    return copy.deepcopy(checkpoint.header), cfg, want


@case("Granite 4.0 350M: the parameters transformers counts, the bytes on disk, the KV cache",
      public=True)  # fmt: skip
def granite_350m(ctx):
    header, cfg, want = small_model(ctx)
    got = report(ctx, header, cfg, "size_report(granite-4.0-350m)")
    for key in ("parameters", "tensor_bytes", "kv_bytes_per_token"):
        check.equal(got[key], want[key], label=f"size_report(granite-4.0-350m)[{key!r}]")


# Rejects name_based_quantization: only this checkpoint has matrices that are not named
# *_proj (shared_mlp.input_linear, shared_mlp.output_linear).
@case("Granite 4.0 350M: the 8-bit prediction is the size of the file mlx-lm writes")
def granite_350m_quantized(ctx):
    header, cfg, want = small_model(ctx)
    got = report(ctx, header, cfg, "size_report(granite-4.0-350m)")
    check.equal(
        got["quantized_bytes"],
        want["quantized_bytes"],
        label="size_report(granite-4.0-350m)['quantized_bytes']",
    )
