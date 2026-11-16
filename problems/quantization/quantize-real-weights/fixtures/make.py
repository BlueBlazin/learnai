"""Regenerate the fixtures of quantize-real-weights.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/quantization/quantize-real-weights/fixtures/make.py

Needs the small house model on disk (``learnai assets fetch granite-4.0-350m``) and the
``models`` extra. The reference implementation is MLX: ``mx.quantize`` and
``mx.dequantize``, which is what ``mlx_lm.convert -q`` runs on every matrix. Error
metrics are computed here in float64 from what MLX returns.

``mx.quantize`` is run on the **CPU** device. On the GPU, MLX 0.32.4 quantizes a group
whose values are all negative differently (it behaves as if the group's maximum were
0, and wastes most of the codes); the CPU kernel follows the published formula, which
is what the statement specifies. Granite's weights contain no such group.

- ``tiny.npz``: for the cases that need no download. Six matrices of shape (6, 256)
  with awkward groups (constant, all zero, one-sided, an outlier) and what MLX makes
  of them at four settings; and the report of a small random "model" at three
  settings. The matrices themselves are not stored: ``matrices`` and ``tiny_model``
  below and their copies in ``tests.py`` draw them from the same seeds, and the file
  records checksums.
- ``granite-4.0-350m.npz``: the report for the real model at 8 and 4 bits (group size
  64), the dequantized values at sampled rows of three matrices, and the KL divergence
  of the quantized model's next-token distributions from the original's on one
  prompt, computed with ``transformers``.

``mx.quantize`` accepts group sizes 32, 64, and 128 only. The statement's worked
example uses a group size of 4; it is checked here with ``affine``, a numpy
transcription of the formula that is asserted to reproduce MLX on every other input.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learnai import assets
from learnai.kit.realmodel import SMALL
from learnai.problem import find_root

HERE = Path(__file__).parent
FLOAT_BITS = 16
SETTINGS = [(2, 32), (4, 64), (8, 128), (8, 32)]  # (bits, group_size) for the matrices
REPORTS = [(4, 64), (8, 64), (4, 32)]  # for the small model
REAL = [(8, 64), (4, 64)]
PROMPT = "The capital of France is Paris. The capital of Japan is"
SAMPLED = {
    "model.embed_tokens.weight": [0, 27286, 100257],
    "model.layers.3.self_attn.q_proj.weight": [0, 511, 1023],
    "model.layers.22.self_attn.v_proj.weight": [5, 255],
}
EXAMPLE = np.array([[-1.0, -0.4, 0.2, 0.5, -0.7, -0.1, 0.3, 0.6]], dtype=np.float32)
EXAMPLE_HAT = [[-1.0, -0.5, 0.0, 0.5, -0.7, 0.0, 0.35, 0.35]]


def affine(w: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """The formula of the statement, in float32 numpy."""
    out, n = w.shape
    levels = np.float32(2**bits - 1)
    groups = w.reshape(out, n // group_size, group_size)
    lo, hi = groups.min(-1, keepdims=True), groups.max(-1, keepdims=True)
    from_low = np.abs(lo) > np.abs(hi)
    step = np.maximum((hi - lo) / levels, np.float32(1e-7))
    scale = np.where(from_low, step, -step)
    edge = np.where(from_low, lo, hi)
    q0 = np.round(edge / scale)
    scale = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scale)
    bias = np.where(q0 != 0, edge, np.float32(0))
    codes = np.clip(np.round((groups - bias) / scale), 0, levels)
    return (codes * scale + bias).reshape(out, n).astype(np.float32)


def mlx_round_trip(w: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    packed, scales, biases = mx.quantize(mx.array(w), group_size=group_size, bits=bits)
    out = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
    result = np.array(out)
    assert np.abs(result - affine(w, bits, group_size)).max() < 1e-6 * max(1.0, np.abs(w).max())
    return result


def matrices() -> dict[str, np.ndarray]:
    """Six (6, 256) matrices with awkward groups. Keep in step with tests.py."""
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
    """A small random "model" with Granite's names. Keep in step with tests.py."""
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


def checksum(arrays: dict[str, np.ndarray]) -> float:
    return float(sum(np.abs(w.astype(np.float64)).sum() for w in arrays.values()))


def report(weights: dict[str, np.ndarray], bits: int, group_size: int, round_trip) -> dict:
    """The report of the statement, in float64, from `round_trip`'s matrices."""
    names, sqnr = [], []
    signal_total = noise_total = 0.0
    stored_bits = parameters = 0
    for name, w in weights.items():
        parameters += w.size
        if w.ndim != 2:
            stored_bits += FLOAT_BITS * w.size
            continue
        w_hat = round_trip(name, w, bits, group_size)
        wide = w.astype(np.float64)
        signal = float((wide**2).sum())
        noise = float(((wide - w_hat) ** 2).sum())
        names.append(name)
        sqnr.append(10 * np.log10(signal / noise))
        signal_total += signal
        noise_total += noise
        stored_bits += bits * w.size + 2 * FLOAT_BITS * (w.size // group_size)
    order = np.argsort(np.array(sqnr), kind="stable")
    gap = np.diff(np.array(sqnr)[order][:4]).min()
    assert gap > 1e-3, f"the worst layers are too close to rank: {gap}"
    return {
        "names": names,
        "sqnr_db": np.array(sqnr),
        "model_sqnr_db": 10 * np.log10(signal_total / noise_total),
        "bits_per_weight": stored_bits / parameters,
        "worst": [names[i] for i in order[:3]],
        "mean_of_db": float(np.mean(sqnr)),
    }


def save_report(out: dict, tag: str, r: dict) -> None:
    out[f"sqnr_{tag}"] = r["sqnr_db"]
    out[f"model_sqnr_{tag}"] = np.array(r["model_sqnr_db"])
    out[f"bpw_{tag}"] = np.array(r["bits_per_weight"])
    out[f"worst_{tag}"] = np.array(r["worst"])


def tiny() -> None:
    assert np.abs(affine(EXAMPLE, 2, 4) - np.array(EXAMPLE_HAT)).max() < 1e-6
    out: dict[str, np.ndarray] = {}
    mats = matrices()
    for bits, group_size in SETTINGS:
        for name, w in mats.items():
            out[f"hat_{name}_{bits}_{group_size}"] = mlx_round_trip(w, bits, group_size)
    out["checksum_matrices"] = np.array(checksum(mats))
    model = tiny_model()
    out["names"] = np.array([name for name, w in model.items() if w.ndim == 2])
    for bits, group_size in REPORTS:
        r = report(model, bits, group_size, lambda _, w, b, g: mlx_round_trip(w, b, g))
        save_report(out, f"{bits}_{group_size}", r)
        print(f"tiny model, {bits} bits, groups of {group_size}: model SQNR "
              f"{r['model_sqnr_db']:.3f} dB (mean of dB {r['mean_of_db']:.3f}), "
              f"{r['bits_per_weight']:.4f} bpw, worst {r['worst']}")  # fmt: skip
    out["checksum_model"] = np.array(checksum(model))
    np.savez_compressed(HERE / "tiny.npz", **out)
    print(f"wrote tiny.npz ({(HERE / 'tiny.npz').stat().st_size} bytes)")


def granite() -> None:
    directory = assets.resolve(SMALL, assets.models_dir(find_root()))
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    state = model.state_dict()
    # The checkpoint's tensors: the tied head is the embedding, stored once.
    # Copies: the state dict is overwritten below, and .numpy() shares its memory.
    weights = {k: v.numpy().copy() for k, v in state.items() if k != "lm_head.weight"}
    assert len(weights) == 226
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0].double(), dim=-1)
    out: dict[str, np.ndarray] = {
        "revision": np.array(SMALL.revision),
        "ids": ids[0].numpy().astype(np.int32),
        "names": np.array([name for name, w in weights.items() if w.ndim == 2]),
    }
    for bits, group_size in REAL:
        tag = f"{bits}_{group_size}"
        hats: dict[str, np.ndarray] = {}

        def round_trip(name, w, b, g, hats=hats):
            hats[name] = mlx_round_trip(w, b, g)
            return hats[name]

        r = report(weights, bits, group_size, round_trip)
        save_report(out, tag, r)
        for name, rows in SAMPLED.items():
            out[f"hat_{tag}_{name}"] = hats[name][rows]
        # The model with every matrix replaced by its quantized version.
        with torch.no_grad():
            for name, w_hat in hats.items():
                state[name].copy_(torch.from_numpy(w_hat))
            logq = torch.log_softmax(model(ids).logits[0].double(), dim=-1)
            for name, w in weights.items():
                state[name].copy_(torch.from_numpy(w))
        kl = float((logp.exp() * (logp - logq)).sum(-1).mean())
        same = int((logp.argmax(-1) == logq.argmax(-1)).sum())
        out[f"kl_{tag}"] = np.array(kl)
        print(f"granite, {bits} bits, groups of {group_size}: model SQNR "
              f"{r['model_sqnr_db']:.3f} dB (mean of dB {r['mean_of_db']:.3f}), "
              f"{r['bits_per_weight']:.5f} bpw, worst {r['worst']}, per-layer "
              f"{r['sqnr_db'].min():.2f} to {r['sqnr_db'].max():.2f} dB, KL {kl:.6f}, "
              f"same argmax at {same} of {ids.shape[1]} positions")  # fmt: skip
    out["sampled"] = np.array(json.dumps(SAMPLED))
    np.savez_compressed(HERE / "granite-4.0-350m.npz", **out)
    print(f"wrote granite-4.0-350m.npz ({(HERE / 'granite-4.0-350m.npz').stat().st_size} bytes)")


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    tiny()
    granite()
