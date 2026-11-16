"""Tests for quantize-real-weights.

Every expected number comes from MLX (`mx.quantize` and `mx.dequantize`, which is what
`mlx_lm.convert -q` runs), with the error metrics computed in float64 by
`fixtures/make.py`. The reference solution is never the oracle.

- The cases marked `assets=[]` use six small matrices with awkward groups and a small
  random "model" with Granite's tensor names. They are drawn here from the same seeds
  as in `make.py`; `tiny.npz` stores what MLX made of them.
- The other cases quantize all 226 tensors of the real granite-4.0-350m (about a
  second per pass), and one of them runs the quantized model through the kit's
  reference forward pass to measure how far its predictions moved.

Dequantized values are compared with `close_to`, which allows up to 2 percent of the
values to sit on a neighboring grid point (never further than about one step away).
Real bfloat16 weights land exactly between two codes surprisingly often, so a quotient
that is one bit off rounds a tie the other way; this happens, for example, when JAX
divides by a Python scalar, which it does by multiplying with the reciprocal. A
flipped tie leaves the size of the error unchanged, and the SQNR with it.
"""

import json
import math

import numpy as np

from learnai.judge import case, check
from learnai.kit import RealModel

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
QD, REPORT = "quantize_dequantize", "quantize_report"
DB_TOL = {"rtol": 1e-3, "atol": 1e-3}  # the tolerance stated in Constraints
KL_LIMIT = {8: 0.01, 4: 2.5}  # nats; stated in Constraints
EXAMPLE = np.array([[-1.0, -0.4, 0.2, 0.5, -0.7, -0.1, 0.3, 0.6]], dtype=np.float32)
EXAMPLE_HAT = np.array([[-1.0, -0.5, 0.0, 0.5, -0.7, 0.0, 0.35, 0.35]], dtype=np.float32)
_cache: dict[str, object] = {}


def matrices() -> dict[str, np.ndarray]:
    """Six (6, 256) matrices with awkward groups. Keep in step with make.py."""
    rng = np.random.default_rng(170)
    out = {}
    for index, std in enumerate((0.02, 0.05, 0.3, 1.0, 3.0, 0.1)):
        w = (rng.standard_normal((6, 256)) * std).astype(np.float32)
        w[0, :128] = 0.0  # all zero
        w[1, 128:256] = np.float32(0.25 * std)  # constant, not zero
        w[2, :128] = np.abs(w[2, :128]) + np.float32(std)  # all positive, away from zero
        w[3, 128:] = -np.abs(w[3, 128:]) - np.float32(2 * std)  # all negative
        w[4, 7] = np.float32(40 * std)  # one outlier stretches its group
        out[f"m{index}"] = w
    return out


def tiny_model() -> dict[str, np.ndarray]:
    """A small random "model" with Granite's names. Keep in step with make.py."""
    rng = np.random.default_rng(171)
    d = 128

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def heavy(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_t(3, (rows, cols)) * std).astype(np.float32)

    def flat(rows: int, cols: int, limit: float) -> np.ndarray:
        return rng.uniform(-limit, limit, (rows, cols)).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, d).astype(np.float32)

    weights = {"model.embed_tokens.weight": normal(96, d, 0.05)}
    for i in range(2):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(d, d, 0.03)
        weights[p + "self_attn.k_proj.weight"] = heavy(64, d, 0.03 * (i + 1))
        weights[p + "self_attn.v_proj.weight"] = heavy(64, d, 0.02)
        weights[p + "self_attn.o_proj.weight"] = flat(d, d, 0.04)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * d, d, 0.02 * (i + 1))
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, d, 0.5)
    weights["model.norm.weight"] = gain()
    weights["lm_head.weight"] = flat(96, d, 0.08)
    return weights


def setup(ctx, which):
    """The seeded inputs of the no-download cases, checked against the fixture."""
    if which not in _cache:
        arrays = matrices() if which == "matrices" else tiny_model()
        total = float(sum(np.abs(w.astype(np.float64)).sum() for w in arrays.values()))
        want = float(ctx.fixture("tiny.npz")[f"checksum_{which}"])
        check.that(
            abs(total - want) < 1e-6 * total,
            "test setup: the random inputs differ from the ones the fixture was made with "
            "(numpy changed its generator?); rerun fixtures/make.py",
        )
        _cache[which] = arrays
    return _cache[which]


def close_to(got, want, w, bits, group_size, label):
    """`got` equals MLX's dequantized `want`, up to a few ties rounded the other way."""
    check.that(isinstance(got, np.ndarray), f"{label}: returned a {type(got).__name__}")
    check.shape(got, want.shape, label=label)
    check.that(got.dtype == np.float32, f"{label}: got dtype {got.dtype}, not float32")
    check.finite(got, label=label)
    diff = np.abs(got.astype(np.float64) - want)
    groups = w.reshape(w.shape[0], -1, group_size)
    step = (groups.max(-1) - groups.min(-1)).astype(np.float64) / (2**bits - 1)
    step = np.repeat(step, group_size, axis=1).reshape(w.shape)  # per value: its group's step
    # Float noise is far below a step; a value on another grid point is a whole step off.
    differing = diff > 0.02 * step + 1e-6 * np.abs(want) + 1e-9
    # No value may be off by more than about one step of its group.
    bound = 2.0 * step + 1e-6
    worst = np.unravel_index(int(np.argmax(diff - bound)), diff.shape)
    detail = (
        f"{int(differing.sum())} of {diff.size} values differ; largest difference "
        f"{diff.max():.3g} at index {list(map(int, np.unravel_index(int(diff.argmax()), diff.shape)))}"
        f"\nexpected (first row, first 8) = {want[0, :8]}\ngot      (first row, first 8) = {got[0, :8]}"
    )
    check.that(
        bool((diff <= bound).all()),
        f"{label}: a value is further than one quantization step from the expected "
        f"result (at index {list(map(int, worst))})",
        detail,
    )
    check.that(
        differing.mean() <= 0.02,
        f"{label}: {differing.mean():.1%} of the values differ from the expected result",
        detail,
    )


def round_trip(ctx, w, bits, group_size, want, label):
    close_to(ctx.call(QD, w, bits, group_size), want, w, bits, group_size, label)


def check_report(ctx, weights, bits, group_size, fixture, tag, label):
    """`weights` is what the solution receives (a dict of arrays, or kit Weights)."""
    got = ctx.call(REPORT, weights, bits, group_size)
    check.that(isinstance(got, dict), f"{label}: must return a dict")
    check.equal(
        sorted(got), ["bits_per_weight", "model_sqnr_db", "sqnr_db", "worst"],
        label=f"{label}: keys",
    )  # fmt: skip
    names = [str(n) for n in fixture["names"]]
    per_layer = got["sqnr_db"]
    check.that(isinstance(per_layer, dict), f'{label}: "sqnr_db" must be a dict')
    check.that(
        all(type(v) is float for v in per_layer.values())
        and type(got["model_sqnr_db"]) is float
        and type(got["bits_per_weight"]) is float,
        f"{label}: every number in the report must be a Python float",
        f"types: { {type(v).__name__ for v in per_layer.values()} }, "
        f"{type(got['model_sqnr_db']).__name__}, {type(got['bits_per_weight']).__name__}",
    )
    missing = [n for n in names if n not in per_layer]
    extra = [n for n in per_layer if n not in names]
    check.that(
        not missing and not extra,
        f'{label}: "sqnr_db" has {len(per_layer)} entries, which are not the matrices of '
        f"the model ({len(missing)} missing, {len(extra)} that are not matrices)",
        f"missing: {missing[:5]}\nnot expected: {extra[:5]}",
    )
    check.allclose(
        np.array([per_layer[n] for n in names]), fixture[f"sqnr_{tag}"], **DB_TOL,
        label=f"{label}: per-layer SQNR in dB (in the order of the tensors)",
    )  # fmt: skip
    check.allclose(got["model_sqnr_db"], float(fixture[f"model_sqnr_{tag}"]), **DB_TOL,
                   label=f'{label}: "model_sqnr_db"')  # fmt: skip
    check.allclose(got["bits_per_weight"], float(fixture[f"bpw_{tag}"]), rtol=1e-6, atol=1e-6,
                   label=f'{label}: "bits_per_weight"')  # fmt: skip
    check.that(
        isinstance(got["worst"], list) and all(isinstance(n, str) for n in got["worst"]),
        f'{label}: "worst" must be a list of tensor names',
    )
    check.equal(got["worst"], [str(n) for n in fixture[f"worst_{tag}"]], label=f'{label}: "worst"')
    return got


# ---------------------------------------------------------------------------------
# Small inputs: no download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: 2 bits, two groups of 4", public=True, assets=[])
def worked_example(ctx):
    got = ctx.call(QD, EXAMPLE, 2, 4)
    check.shape(got, (1, 8), label="quantize_dequantize(example)")
    check.allclose(got, EXAMPLE_HAT, rtol=1e-5, atol=1e-6, dtype=True,
                   label="quantize_dequantize(example)")  # fmt: skip


@case("random matrices at 4 bits, groups of 64, against MLX", public=True, assets=[])
def against_mlx(ctx):
    fixture = ctx.fixture("tiny.npz")
    for name, w in setup(ctx, "matrices").items():
        round_trip(ctx, w, 4, 64, fixture[f"hat_{name}_4_64"],
                   f"quantize_dequantize({name}, bits=4, group_size=64)")  # fmt: skip


# Rejects no_scale_floor: the first matrix row has an all-zero and a constant group.
@case("awkward groups: all zero, constant, one-sided, an outlier; 2 and 8 bits",
      public=True, assets=[])  # fmt: skip
def awkward_groups(ctx):
    fixture = ctx.fixture("tiny.npz")
    mats = setup(ctx, "matrices")
    for bits, group_size in ((2, 32), (8, 128), (8, 32)):
        for name in ("m0", "m3", "m4"):
            round_trip(ctx, mats[name], bits, group_size, fixture[f"hat_{name}_{bits}_{group_size}"],
                       f"quantize_dequantize({name}, bits={bits}, group_size={group_size})")  # fmt: skip
    zeros = ctx.call(QD, mats["m0"], 4, 64)
    check.that(
        bool((zeros[0, :128] == 0).all()),
        "quantize_dequantize: an all-zero group did not come back as zeros",
    )


# Rejects bpw_without_overhead, skips_embedding, mean_of_decibels, worst_is_best,
# amplitude_decibels, counts_only_matrices.
@case("the report of a small model: 4 bits, groups of 64", public=True, assets=[])
def small_report(ctx):
    fixture = ctx.fixture("tiny.npz")
    check_report(ctx, setup(ctx, "model"), 4, 64, fixture, "4_64", "quantize_report(small model)")


@case("the report of a small model at 8 bits, and with groups of 32", assets=[])
def small_report_settings(ctx):
    fixture = ctx.fixture("tiny.npz")
    model = setup(ctx, "model")
    check_report(ctx, model, 8, 64, fixture, "8_64", "quantize_report(bits=8, group_size=64)")
    check_report(ctx, model, 4, 32, fixture, "4_32", "quantize_report(bits=4, group_size=32)")


@case("quantize_report does not depend on the order of the tensors", assets=[])
def order_independent(ctx):
    fixture = ctx.fixture("tiny.npz")
    model = setup(ctx, "model")
    shuffled = {name: model[name] for name in sorted(model, reverse=True)}
    check_report(ctx, shuffled, 4, 64, fixture, "4_64", "quantize_report(tensors in another order)")


# ---------------------------------------------------------------------------------
# granite-4.0-350m: 169 matrices, 352 M weights.
# ---------------------------------------------------------------------------------


def granite(ctx):
    fixture = ctx.fixture("granite-4.0-350m.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    model = RealModel(ctx.asset(ASSET))
    key = f"weights_{ctx.framework}"
    if key not in _cache:
        # Bound to the framework once: each tensor is converted the first time it is used.
        _cache[key] = model.weights(ctx.fw)
    return model, _cache[key], fixture


def real_report(ctx, bits):
    _, weights, fixture = granite(ctx)
    got = check_report(ctx, weights, bits, 64, fixture, f"{bits}_64",
                       f"quantize_report(granite, bits={bits}, group_size=64)")  # fmt: skip
    ctx.metric("model_sqnr_db", round(got["model_sqnr_db"], 2), "dB", higher_is_better=True)
    ctx.metric("bits_per_weight", round(got["bits_per_weight"], 4), "bits")


# Rejects bits_in_int32 (3.0e9 stored bits do not fit in 32 bits; the small model's do).
@case("Granite 4.0 350M at 8 bits, groups of 64: per-layer SQNR, 8.501 bits per weight",
      public=True)  # fmt: skip
def real_8bit(ctx):
    real_report(ctx, 8)


@case("Granite 4.0 350M at 4 bits, groups of 64")
def real_4bit(ctx):
    real_report(ctx, 4)


@case("Granite 4.0 350M: dequantized rows of three matrices against MLX, 8 and 4 bits")
def real_values(ctx):
    model, _, fixture = granite(ctx)
    plain = model.weights()
    for name, rows in json.loads(str(fixture["sampled"])).items():
        w = plain.numpy(name)[rows]
        for bits in (8, 4):
            round_trip(ctx, w, bits, 64, fixture[f"hat_{bits}_64_{name}"],
                       f"quantize_dequantize(rows of a real matrix, bits={bits})")  # fmt: skip


class Quantized:
    """Stands in for a RealModel whose every matrix went through the solution.

    The kit's RealLM asks for one tensor at a time, by name, in the order of the
    forward pass; only the embedding (also the output head) is asked for twice.
    """

    def __init__(self, ctx, model, bits, group_size):
        self.ctx, self.config, self.path = ctx, model.config, model.path
        self.plain = model.weights()
        self.bits, self.group_size = bits, group_size
        self.kept: dict[str, np.ndarray] = {}

    def weights(self):
        return self

    def numpy(self, name):
        w = self.plain.numpy(name)
        if w.ndim != 2:
            return w
        if name in self.kept:
            return self.kept[name]
        w_hat = np.asarray(self.ctx.call(QD, w, self.bits, self.group_size), dtype=np.float32)
        check.shape(w_hat, w.shape, label="quantize_dequantize(a real matrix)")
        if name == "model.embed_tokens.weight":
            self.kept[name] = w_hat
        return w_hat


def real_kl(ctx, bits):
    from learnai.kit import RealLM

    model, _, fixture = granite(ctx)
    ids = [int(t) for t in fixture["ids"]]

    def log_probs(source):
        logits = np.asarray(RealLM(source)(ids)[0], dtype=np.float64)
        top = logits.max(-1, keepdims=True)
        return logits - top - np.log(np.exp(logits - top).sum(-1, keepdims=True))

    if "logp" not in _cache:
        _cache["logp"] = log_probs(model)
    logp = _cache["logp"]
    logq = log_probs(Quantized(ctx, model, bits, 64))
    check.finite(logq, label="the logits of the quantized model")
    kl = float((np.exp(logp) * (logp - logq)).sum(-1).mean())
    check.that(
        not math.isnan(kl) and kl > 1e-7,
        "the quantized model predicts exactly what the original does: nothing was quantized",
    )
    check.at_most("logit_kl", round(kl, 5), KL_LIMIT[bits], unit="nats")


@case("Granite 4.0 350M at 8 bits: the quantized model's predictions stay within "
      "0.01 nats (KL) of the original's", public=True, time_limit_s=60)  # fmt: skip
def real_kl_8bit(ctx):
    real_kl(ctx, 8)


@case("Granite 4.0 350M at 4 bits: the predictions move, but by less than 2.5 nats",
      time_limit_s=60)  # fmt: skip
def real_kl_4bit(ctx):
    real_kl(ctx, 4)
