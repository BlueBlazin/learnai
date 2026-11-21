"""Regenerate the fixture of layer-plan-from-config.

    uv run --project python --extra models \
        python problems/real-hybrid-model/layer-plan-from-config/fixtures/make.py

Needs the ``models`` extra (``transformers`` and torch) and no weights. Every expected
plan in ``cases.json`` is read off the objects ``transformers`` builds from the
configuration, never off this problem's reference solution:

- the schedule, the MLP kinds, the indexer modes, and the key-value sharing modes from
  ``Qwen3_5TextConfig``, ``GlmMoeDsaConfig``, and ``Gemma4TextConfig`` (``layer_types``,
  ``mlp_layer_types``, ``indexer_types``, ``kv_sharing_modes``);
- the head width from the resolved configuration (``head_dim``, ``qk_head_dim``, and
  for Gemma 4 the per-layer view ``per_layer_config[i].head_dim``);
- the number of rotating channels from the frequency table of the model's rotary
  embedding module: twice the number of its nonzero inverse frequencies.

``real-configs.json`` holds reduced copies of three published files, ``config.json`` of
``Qwen/Qwen3.5-0.8B`` (revision 2fc06364715b967f1860aea9cf38778875588b17),
``zai-org/GLM-5.3``, and ``google/gemma-4-E2B-it``, as they were on 2026-11-21. Reduced
means: the vision and audio towers, token ids, dtypes, dropout and initializer
settings, and GLM's ``quantization_config`` are dropped; every key a layer plan reads,
and the main sizes, are kept with their published values, including the nesting under
``text_config`` and the explicit per-layer lists.

The other configurations are variations written here: the real files with their
explicit lists removed, short configurations that lean on defaults, explicit lists the
rules would not produce, and seeded random sweeps.
"""

import copy
import json
from pathlib import Path

import numpy as np
from transformers import Gemma4TextConfig, GlmMoeDsaConfig, Qwen3_5TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

HERE = Path(__file__).parent
REAL = json.loads((HERE / "real-configs.json").read_text(encoding="utf-8"))
FULL, LINEAR, SLIDING = "full_attention", "linear_attention", "sliding_attention"


# ---------------------------------------------------------------------------------
# The plan, as the shipped classes resolve it.
# ---------------------------------------------------------------------------------


def rotating(inv_freq) -> int:
    """Channels of a head that turn: two per nonzero frequency."""
    return 2 * int((inv_freq != 0).sum())


def entry(mixer, mlp="dense", indexer=None, rope=None, kv=None) -> dict:
    return {"mixer": mixer, "mlp": mlp, "indexer": indexer, "rope": rope, "kv": kv}


def shipped_plan(config: dict) -> list[dict]:
    text = config.get("text_config", config)
    family = text["model_type"]
    kwargs = {key: value for key, value in copy.deepcopy(text).items() if key != "model_type"}
    if family == "qwen3_5_text":
        cfg = Qwen3_5TextConfig(**kwargs)
        rope = {
            "type": cfg.rope_parameters["rope_type"],
            "theta": cfg.rope_parameters["rope_theta"],
            "head_dim": cfg.head_dim,
            "rotary_dim": rotating(Qwen3_5TextRotaryEmbedding(cfg).inv_freq),
        }
        # Qwen3_5DecoderLayer hands position embeddings to full-attention layers only.
        return [entry(kind, rope=dict(rope) if kind == FULL else None) for kind in cfg.layer_types]
    if family == "glm_moe_dsa":
        cfg = GlmMoeDsaConfig(**kwargs)
        rope = {
            "type": cfg.rope_parameters["rope_type"],
            "theta": cfg.rope_parameters["rope_theta"],
            "head_dim": cfg.qk_head_dim,
            "rotary_dim": rotating(GlmMoeDsaRotaryEmbedding(cfg).inv_freq),
        }
        return [
            entry(cfg.layer_types[i], cfg.mlp_layer_types[i], cfg.indexer_types[i], dict(rope))
            for i in range(cfg.num_hidden_layers)
        ]
    if family == "gemma4_text":
        cfg = Gemma4TextConfig(**kwargs)
        assert not cfg.enable_moe_block
        tables = Gemma4TextRotaryEmbedding(cfg)
        modes = cfg.kv_sharing_modes
        plan = []
        for i, kind in enumerate(cfg.layer_types):
            params = cfg.rope_parameters[kind]
            rope = {
                "type": params["rope_type"],
                "theta": params["rope_theta"],
                "head_dim": cfg.per_layer_config[i].head_dim,
                "rotary_dim": rotating(getattr(tables, f"{kind}_inv_freq")),
            }
            plan.append(entry(kind, rope=rope, kv=modes[i]))
        return plan
    raise ValueError(family)


# ---------------------------------------------------------------------------------
# The configurations.
# ---------------------------------------------------------------------------------


def without(config: dict, *keys: str) -> dict:
    """A copy of a real file with keys of its decoder settings removed."""
    out = copy.deepcopy(config)
    text = out.get("text_config", out)
    for key in keys:
        del text[key]
    return out


def qwen(n: int, head_dim: int = 64, *, theta: float = 1e7, factor=None, **extra) -> dict:
    rope = {"rope_type": "default", "rope_theta": theta}
    if factor is not None:
        rope["partial_rotary_factor"] = factor
    return {"model_type": "qwen3_5_text", "num_hidden_layers": n, "head_dim": head_dim,
            "rope_parameters": rope, **extra}  # fmt: skip


def glm(n: int, *, nope: int = 24, rotary: int = 8, theta: float = 8e6, **extra) -> dict:
    # `head_dim` is in the published file with the value of qk_nope_head_dim.
    return {"model_type": "glm_moe_dsa", "num_hidden_layers": n, "head_dim": nope,
            "qk_nope_head_dim": nope, "qk_rope_head_dim": rotary,
            "rope_parameters": {"rope_type": "default", "rope_theta": theta}, **extra}  # fmt: skip


def gemma(kinds: list[str], head_dim: int = 256, *, full=None, sliding=None, **extra) -> dict:
    full = full or {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6}
    sliding = sliding or {"rope_type": "default", "rope_theta": 1e4}
    return {"model_type": "gemma4_text", "num_hidden_layers": len(kinds), "head_dim": head_dim,
            "layer_types": kinds, "rope_parameters": {FULL: full, SLIDING: sliding},
            **extra}  # fmt: skip


def wrapped(text: dict, outer: str) -> dict:
    return {"model_type": outer, "text_config": text}


def named_cases() -> dict[str, dict]:
    s, f = SLIDING, FULL
    return {
        # The three published files.
        "qwen_real": REAL["qwen3.5-0.8b"],
        "glm_real": REAL["glm-5.3"],
        "gemma_real": REAL["gemma-4-e2b"],
        # The same files without their explicit lists: the rules must regenerate them.
        "qwen_real_no_list": without(REAL["qwen3.5-0.8b"], "layer_types"),
        "glm_real_no_lists": without(REAL["glm-5.3"], "indexer_types", "mlp_layer_types"),
        # Explicit lists that no rule produces.
        "qwen_explicit": wrapped(
            qwen(6, full_attention_interval=4, factor=0.25,
                 layer_types=[FULL, LINEAR, LINEAR, FULL, FULL, LINEAR]),
            "qwen3_5",
        ),
        "glm_explicit": glm(
            8, first_k_dense_replace=3, index_topk_freq=4, index_skip_topk_offset=3,
            index_topk_pattern="FFFFSSSS",
            indexer_types=["full", "shared", "full", "full", "shared", "shared", "shared", "full"],
            mlp_layer_types=["dense", "sparse", "dense", "sparse", "sparse", "sparse", "dense",
                             "sparse"],
        ),
        "glm_pattern_string": glm(
            8, index_topk_freq=4, index_skip_topk_offset=3, index_topk_pattern="FSSFSFSS",
            indexer_types=None, mlp_layer_types=None, first_k_dense_replace=2,
        ),
        "glm_pattern_list": glm(
            5, index_topk_freq=2, index_skip_topk_offset=1,
            index_topk_pattern=["full", "full", "shared", "full", "shared"],
        ),
        # Short configurations: the defaults.
        "qwen_short": qwen(7, 64, theta=1e4),
        "qwen_interval_3": wrapped(
            qwen(7, 32, factor=0.5, full_attention_interval=3, layer_types=None), "qwen3_5"
        ),
        "glm_short": glm(9),
        "glm_no_offset": glm(9, index_topk_freq=3, first_k_dense_replace=1),
        "glm_freq_zero": glm(7, index_topk_freq=0, index_skip_topk_offset=3),
        "gemma_no_sharing": wrapped(gemma([s, s, f, s, s, f], 128), "gemma4"),
        "gemma_zero_shared": gemma([s, f, s, f], num_kv_shared_layers=0, global_head_dim=384),
        # Where "store" goes, and the two RoPE conventions.
        "gemma_store_far_back": gemma(
            [s, f, s, s, s, s, f, s, s, f], num_kv_shared_layers=4, global_head_dim=512
        ),
        "gemma_odd_product": gemma(
            [s, s, f, s, f], 12, num_kv_shared_layers=2, global_head_dim=20,
            full={"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 5e5},
            sliding={"rope_type": "default", "rope_theta": 2e4},
        ),
        # Edges, for hidden cases.
        "qwen_all_full": qwen(5, 16, full_attention_interval=1, factor=1.0),
        "qwen_too_short": qwen(3, 128, full_attention_interval=4),
        "glm_offset_zero": glm(9, index_topk_freq=4, index_skip_topk_offset=0),
        "glm_offset_one": glm(9, index_topk_freq=3, index_skip_topk_offset=1),
        "glm_offset_past_end": glm(6, index_topk_freq=2, index_skip_topk_offset=9),
        "glm_all_dense": glm(4, first_k_dense_replace=7, index_topk_freq=2),
        "glm_no_dense": glm(4, first_k_dense_replace=0, index_topk_freq=2, index_skip_topk_offset=2),
        "gemma_all_full": gemma([f, f, f, f, f], 64, num_kv_shared_layers=3, global_head_dim=128),
        "gemma_one_unshared_each": gemma([s, f, s, s, f, s, f], 32, num_kv_shared_layers=5),
        "gemma_swapped_types": gemma(
            [s, f, s, f], 24, num_kv_shared_layers=1, global_head_dim=40,
            full={"rope_type": "default", "rope_theta": 3e5},
            sliding={"rope_type": "proportional", "partial_rotary_factor": 0.3, "rope_theta": 1e4},
        ),
        "gemma_proportional_no_factor": gemma(
            [s, f], 16, global_head_dim=48, full={"rope_type": "proportional", "rope_theta": 1e6}
        ),
        # The same three rules again, for hidden cases: a proportional entry without a
        # factor, no global_head_dim, and a stored type that no shared layer reuses.
        "gemma_no_factor_again": gemma(
            [f, s, s, f], 32, global_head_dim=96, num_kv_shared_layers=1,
            full={"rope_type": "proportional", "rope_theta": 2e6},
        ),
        "gemma_no_global_again": gemma([s, f, s, f], 64, num_kv_shared_layers=2),
        "gemma_unreused_store_again": gemma([s, f, s, s, f], 32, num_kv_shared_layers=1),
    }


def sweeps() -> dict[str, dict]:
    """Seeded random configurations of each family, for the hidden sweeps."""
    rng = np.random.default_rng(250)
    out: dict[str, dict] = {}

    def pick(options):
        return options[int(rng.integers(len(options)))]

    for index in range(8):
        n = int(rng.integers(1, 25))
        extra = {}
        if rng.random() < 0.7:
            extra["full_attention_interval"] = int(rng.integers(1, 7))
        if rng.random() < 0.35:
            extra["layer_types"] = [pick([FULL, LINEAR]) for _ in range(n)]
        elif rng.random() < 0.3:
            extra["layer_types"] = None
        text = qwen(n, pick([16, 64, 256]), theta=pick([1e4, 1e6, 1e7]),
                    factor=pick([None, 0.25, 0.5, 1.0]), **extra)  # fmt: skip
        out[f"qwen_sweep_{index}"] = wrapped(text, "qwen3_5") if rng.random() < 0.5 else text

    for index in range(10):
        n = int(rng.integers(1, 25))
        extra = {}
        if rng.random() < 0.8:
            extra["index_topk_freq"] = int(rng.integers(0, 7))
        if rng.random() < 0.8:
            extra["index_skip_topk_offset"] = int(rng.integers(0, 8))
        if rng.random() < 0.8:
            extra["first_k_dense_replace"] = int(rng.integers(0, n + 3))
        choice = rng.random()
        if choice < 0.2:
            extra["index_topk_pattern"] = "".join(pick(["F", "S"]) for _ in range(n))
        elif choice < 0.3:
            extra["index_topk_pattern"] = [pick(["full", "shared"]) for _ in range(n)]
        elif choice < 0.6:
            extra["index_topk_pattern"] = None
        if rng.random() < 0.2:
            extra["indexer_types"] = [pick(["full", "shared"]) for _ in range(n)]
        if rng.random() < 0.2:
            extra["mlp_layer_types"] = [pick(["dense", "sparse"]) for _ in range(n)]
        nope, rotary = pick([(24, 8), (192, 64), (16, 16)])
        out[f"glm_sweep_{index}"] = glm(n, nope=nope, rotary=rotary, theta=pick([1e4, 8e6]), **extra)

    index = 0
    while index < 10:
        n = int(rng.integers(1, 21))
        kinds = [pick([SLIDING, SLIDING, FULL]) for _ in range(n - 1)] + [FULL]
        shared = int(rng.integers(0, n))
        if shared and set(kinds[: n - shared]) != set(kinds):
            continue  # transformers refuses a layer type that has no layer producing its cache
        extra = {}
        if shared or rng.random() < 0.5:
            extra["num_kv_shared_layers"] = shared
        if rng.random() < 0.7:
            extra["global_head_dim"] = pick([20, 48, 128, 512])
        full = {"rope_type": "proportional", "rope_theta": pick([1e6, 5e5])}
        if rng.random() < 0.8:
            full["partial_rotary_factor"] = pick([0.25, 0.3, 0.5, 1.0])
        text = gemma(kinds, pick([12, 32, 256]), full=full, **extra)
        out[f"gemma_sweep_{index}"] = wrapped(text, "gemma4") if rng.random() < 0.5 else text
        index += 1
    return out


def main() -> None:
    configs = {**named_cases(), **sweeps()}
    cases = {}
    for name, config in configs.items():
        cases[name] = {"config": config, "plan": shipped_plan(config)}
    target = HERE / "cases.json"
    target.write_text(json.dumps(cases, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {target.name}: {len(cases)} configurations, {target.stat().st_size} bytes")
    plan = cases["qwen_real"]["plan"]
    print("worked example, layers 0 to 3 of qwen_real:")
    for layer in plan[:4]:
        print("  ", layer)
    print("gemma_real kv:", [layer["kv"] for layer in cases["gemma_real"]["plan"]])
    print("gemma_real layers 3, 4:", cases["gemma_real"]["plan"][3:5])
    print("glm_real layers 0, 3:", plan_rows(cases["glm_real"]["plan"], 0, 3))
    print("glm_real indexers:", "".join(e["indexer"][0] for e in cases["glm_real"]["plan"]))


def plan_rows(plan: list[dict], *rows: int) -> list[dict]:
    return [plan[row] for row in rows]


if __name__ == "__main__":
    main()
