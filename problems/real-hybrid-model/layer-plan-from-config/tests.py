"""Tests for layer-plan-from-config.

Every expected plan is in `fixtures/cases.json`, which `fixtures/make.py` writes by
building the configuration classes of `transformers` (`Qwen3_5TextConfig`,
`GlmMoeDsaConfig`, `Gemma4TextConfig`) and the models' rotary-embedding modules from
each configuration; the reference solution is never the oracle. The three published
files are reduced copies (see `make.py`); the rest are variations of them, short
configurations that lean on defaults, and seeded random sweeps.

Public cases name the configuration in their messages. Hidden cases say only
"layer_plan(config)".
"""

import copy

from learnai.judge import case, check

FN = "layer_plan"
KEYS = {"mixer", "mlp", "indexer", "rope", "kv"}


def run(ctx, name, what=None):
    """Compare the plan of fixture configuration `name` with the shipped one."""
    fixture = ctx.fixture("cases.json")[name]
    label = f"layer_plan({what})" if what else "layer_plan(config)"
    got = ctx.call(FN, copy.deepcopy(fixture["config"]))
    check.that(
        isinstance(got, list) and all(isinstance(entry, dict) for entry in got),
        f"{label}: must return a list of dicts, one per layer",
        f"got {got!r}"[:600],
    )
    check.that(
        all(set(entry) == KEYS for entry in got),
        f"{label}: every entry must have exactly the keys mixer, mlp, indexer, rope, kv",
        f"keys of the entries: {sorted({tuple(sorted(map(str, entry))) for entry in got})}",
    )
    # First each column on its own, so the message says which part of the plan is off.
    if len(got) == len(fixture["plan"]):
        for key in ("mixer", "mlp", "indexer", "kv", "rope"):
            check.equal(
                [entry[key] for entry in got],
                [entry[key] for entry in fixture["plan"]],
                label=f"{label}: the `{key}` of each layer",
            )
    check.equal(got, fixture["plan"], label=label)
    return got


def sweep(ctx, prefix):
    names = [name for name in ctx.fixture("cases.json") if name.startswith(prefix)]
    check.that(len(names) >= 8, "test setup: the fixture has no such sweep")
    for name in names:
        run(ctx, name)


# ---------------------------------------------------------------------------------
# Public: the three published files, then one family of rules per case.
# ---------------------------------------------------------------------------------


@case("the worked example: Qwen3.5 0.8B, 24 layers, attention at layers 3, 7, ..., 23",
      public=True)  # fmt: skip
def worked_example(ctx):
    plan = run(ctx, "qwen_real", "the Qwen3.5 0.8B file")
    # The two rows printed in the statement.
    check.equal(
        plan[0],
        {"mixer": "linear_attention", "mlp": "dense", "indexer": None, "rope": None, "kv": None},
        label="layer 0 of the Qwen3.5 0.8B plan",
    )
    check.equal(
        plan[3],
        {
            "mixer": "full_attention",
            "mlp": "dense",
            "indexer": None,
            "rope": {"type": "default", "theta": 10000000, "head_dim": 256, "rotary_dim": 64},
            "kv": None,
        },
        label="layer 3 of the Qwen3.5 0.8B plan",
    )


# Rejects glm_head_dim_from_file (the file's `head_dim` is 192), sliding_rope_for_full,
# global_head_dim_ignored, and store_last_unshared_only (layers 13 and 14 both store).
@case("GLM-5.3 (78 layers) and Gemma 4 E2B (35 layers, one `store` per layer type)",
      public=True)  # fmt: skip
def other_real_files(ctx):
    run(ctx, "glm_real", "the GLM-5.3 file")
    run(ctx, "gemma_real", "the Gemma 4 E2B file")


# Rejects schedule_from_zero, index_rule_no_clamp, and index_offset_off_by_one: without
# the list, the rule has to give what the file lists.
@case("the Qwen and GLM files with their per-layer lists removed: the rules regenerate them",
      public=True)  # fmt: skip
def lists_removed(ctx):
    run(ctx, "qwen_real_no_list", "the Qwen3.5 file without layer_types")
    run(ctx, "glm_real_no_lists", "the GLM-5.3 file without indexer_types and mlp_layer_types")


# Rejects ignores_explicit_lists, pattern_ignored, and pattern_over_explicit_list.
@case("explicit lists win over the rules; `index_topk_pattern` as a string and as a list",
      public=True)  # fmt: skip
def explicit_lists(ctx):
    run(ctx, "qwen_explicit", "6 Qwen layers with an irregular layer_types")
    run(ctx, "glm_explicit", "8 GLM layers with indexer_types, mlp_layer_types, and a pattern")
    run(ctx, "glm_pattern_string", "8 GLM layers with index_topk_pattern 'FSSFSFSS'")
    run(ctx, "glm_pattern_list", "5 GLM layers with index_topk_pattern as a list")


# Rejects qwen_factor_default_one, missing_offset_zero, and store_when_nothing_shared.
@case("short configurations: every default, a frequency of 0, and no shared layers",
      public=True)  # fmt: skip
def defaults(ctx):
    run(ctx, "qwen_short", "7 Qwen layers, no interval, no partial_rotary_factor")
    run(ctx, "qwen_interval_3", "7 Qwen layers, interval 3, layer_types null")
    run(ctx, "glm_short", "9 GLM layers with no schedule keys at all")
    run(ctx, "glm_no_offset", "9 GLM layers, index_topk_freq 3, no offset")
    run(ctx, "glm_freq_zero", "7 GLM layers, index_topk_freq 0")
    run(ctx, "gemma_no_sharing", "6 Gemma layers, head_dim 128, no num_kv_shared_layers or global_head_dim")
    run(ctx, "gemma_zero_shared", "4 Gemma layers with num_kv_shared_layers 0")


# Rejects proportional_as_default (20 * 0.25 = 5 is odd) and, again,
# store_last_unshared_only (the last full layer that still stores is layer 1). The last
# two configurations are the public witnesses of three rules: a proportional entry
# without a factor rotates the whole head; a layer type stores even when no shared
# layer is of that type; the RoPE type belongs to the entry, not to the layer type.
@case("Gemma 4: where `store` goes, a proportional head of 20, and a missing factor",
      public=True)  # fmt: skip
def gemma_rules(ctx):
    run(ctx, "gemma_store_far_back", "10 Gemma layers, 4 shared")
    run(ctx, "gemma_odd_product", "5 Gemma layers, global_head_dim 20, factor 0.25")
    run(ctx, "gemma_proportional_no_factor", "2 Gemma layers, proportional without a factor")
    run(ctx, "gemma_swapped_types", "4 Gemma layers, 1 shared, proportional sliding layers")


UNKNOWN = [
    {"model_type": "llama", "num_hidden_layers": 2, "head_dim": 64},
    {"num_hidden_layers": 2, "head_dim": 64},
    # The wrapper's own model_type is not the decoder's.
    {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_text", "num_hidden_layers": 2}},
]


@case("an unknown or missing model_type raises ValueError", public=True)
def unknown_family(ctx):
    for config in UNKNOWN:
        check.raises(ValueError, ctx.solution.layer_plan, copy.deepcopy(config),
                     label=f"layer_plan({config})")  # fmt: skip


# ---------------------------------------------------------------------------------
# Hidden: sweeps and edges. Messages name nothing.
# ---------------------------------------------------------------------------------


@case("Qwen3.5: random layer counts, intervals, lists, and rotary factors")
def qwen_sweep(ctx):
    sweep(ctx, "qwen_sweep_")


@case("GLM: random frequencies, offsets, patterns, lists, and dense counts")
def glm_sweep(ctx):
    sweep(ctx, "glm_sweep_")


@case("Gemma 4: random schedules, shared counts, head widths, and factors")
def gemma_sweep(ctx):
    sweep(ctx, "gemma_sweep_")


@case("schedule edges: interval 1, fewer layers than the interval, offsets 0, 1, past the end")
def schedule_edges(ctx):
    for name in ("qwen_all_full", "qwen_too_short", "glm_offset_zero", "glm_offset_one",
                 "glm_offset_past_end", "glm_all_dense", "glm_no_dense"):  # fmt: skip
        run(ctx, name)


@case("Gemma 4 edges: one layer type, one unshared layer per type, defaults again")
def gemma_edges(ctx):
    for name in ("gemma_all_full", "gemma_one_unshared_each", "gemma_no_factor_again",
                 "gemma_no_global_again", "gemma_unreused_store_again"):  # fmt: skip
        run(ctx, name)


@case("other unsupported model types raise ValueError")
def unknown_family_hidden(ctx):
    for config in (
        {"model_type": "qwen3_5", "num_hidden_layers": 4},
        {"model_type": "gemma4", "text_config": {"model_type": "gemma3_text", "layer_types": []}},
        {"model_type": None, "num_hidden_layers": 1},
    ):
        check.raises(ValueError, ctx.solution.layer_plan, copy.deepcopy(config), label=FN)
