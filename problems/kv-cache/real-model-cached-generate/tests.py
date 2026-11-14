"""Tests for real-model-cached-generate.

Every expected token and logit comes from `transformers` (float32, CPU), stored by
`fixtures/make.py`: greedy `generate` for the tokens, and forward passes over whole
sequences, without any cache, for the logits. The reference solution is never the
oracle; it is only the baseline of the speed case.

- The cases marked `assets=[]` run two three-layer Granite models with random weights
  and a 64-token vocabulary: `a` with Granite 4.0's constants and a tied head, `b`
  with other constants, RoPE base 10,000, and an untied head. Their weights are drawn
  here from the same seeds as in `make.py`. Prompts have 6 tokens, or 1.
- The other cases run the real granite-4.0-350m: two prompts of 12 tokens, and one of
  768 tokens, after which the speed case times four decode steps.

The solution's cache is opaque, so `prefill` and `decode_step` are called directly
(`ctx.call` would convert the tensors inside the cache to numpy).
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import RealConfig, RealModel

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
TOL = {"rtol": 1e-3, "atol": 1e-3}  # the tolerance stated in Constraints
MAX_RATIO = 3.0  # the speed limit stated in Constraints

TINY = {"vocab": 64, "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "d_ff": 96, "n_layers": 3}
# The worked example of the statement: model a, the one-token prompt [41].
WORKED_TOKENS = [10, 10, 16, 16, 16, 16, 30, 38]
_cache: dict[str, object] = {}


def tiny_weights(seed: int, tied: bool, embed_std: float) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout. Keep in step with make.py."""
    rng = np.random.default_rng(seed)
    d, f, vocab = TINY["d_model"], TINY["d_ff"], TINY["vocab"]
    q_width = d
    kv_width = d * TINY["n_kv_heads"] // TINY["n_heads"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, d).astype(np.float32)

    weights = {"model.embed_tokens.weight": normal(vocab, d, embed_std)}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(q_width, d, 0.3)
        weights[p + "self_attn.k_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.v_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.o_proj.weight"] = normal(d, q_width, 0.3)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.2)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.3)
    weights["model.norm.weight"] = gain()
    if not tied:
        weights["lm_head.weight"] = normal(vocab, d, 1.0)
    return weights


class Model:
    """Weights in the solution's framework, the config, and the fixture of one model."""

    def __init__(self, ctx, weights, cfg, fixture, key, columns=None):
        self.ctx, self.weights, self.cfg, self.fixture = ctx, weights, cfg, fixture
        self.key = key  # "a_0": model a, prompt 0; "1": real prompt 1
        self.columns = columns  # the vocabulary columns the fixture stores, or all
        self.prompt = fixture[f"prompt_{key}"].astype(np.int64)
        self.tokens = [int(t) for t in fixture[f"tokens_{key}"]]
        self.logits = fixture[f"logits_{key}"]  # row i chose tokens[i]

    def logits_of(self, out, label):
        """Check what prefill or decode_step returned; give back (logits, cache)."""
        ctx = self.ctx
        check.that(
            isinstance(out, tuple) and len(out) == 2,
            f"{label}: must return a tuple (logits, cache)",
            f"got: {type(out).__name__}",
        )
        logits, cache = out
        check.that(
            ctx.fw.is_tensor(logits),
            f"{label}: the logits are a {type(logits).__name__}, not a tensor of the framework",
        )
        got = np.asarray(ctx.fw.to_numpy(logits))
        check.shape(got, (self.cfg.vocab_size,), label=f"{label}: logits")
        check.that(got.dtype == np.float32, f"{label}: the logits have dtype {got.dtype}")
        return got, cache

    def compare(self, got, want, label):
        shown = got if self.columns is None else got[self.columns]
        where = "logits" if self.columns is None else "logits at 64 vocabulary columns"
        check.allclose(shown, want, **TOL, label=f"{label}: {where}")

    def prefill(self, label="prefill"):
        ctx = self.ctx
        out = ctx.solution.prefill(self.weights, self.cfg, ctx.fw.tensor(self.prompt))
        got, cache = self.logits_of(out, label)
        self.compare(got, self.logits[0], label)
        return got, cache

    def step(self, token, cache, want, label):
        out = self.ctx.solution.decode_step(self.weights, self.cfg, int(token), cache)
        got, cache = self.logits_of(out, label)
        self.compare(got, want, label)
        return got, cache

    def stepwise(self, steps=None):
        """prefill, then decode_step fed with the fixture's tokens: logits at every step."""
        got, cache = self.prefill()
        chosen = [int(got.argmax())]
        for i, token in enumerate(self.tokens[: (steps or len(self.tokens)) - 1]):
            got, cache = self.step(token, cache, self.logits[i + 1], f"decode_step {i + 1}")
            chosen.append(int(got.argmax()))
        check.equal(chosen, self.tokens[: len(chosen)], label="the argmax token of every step")
        return cache

    def generate(self, n=None, label="generate"):
        ctx = self.ctx
        n = len(self.tokens) if n is None else n
        got = ctx.solution.generate(self.weights, self.cfg, ctx.fw.tensor(self.prompt), n)
        check.that(
            isinstance(got, list) and all(type(t) is int for t in got),
            f"{label}: must return a list of Python ints",
            f"got: {got!r}"[:300],
        )
        check.equal(got, self.tokens[:n], label=f"{label}: tokens")
        return got

    def branch(self):
        """Two continuations of one cache, then the first one continued."""
        _, start = self.prefill()
        first, other = self.tokens[0], int(self.fixture[f"other_{self.key}"])
        _, after_first = self.step(first, start, self.logits[1], "decode_step(first token)")
        self.step(other, start, self.fixture[f"branch_{self.key}"],
                  "decode_step(another token, from the cache prefill returned)")  # fmt: skip
        self.step(self.tokens[1], after_first, self.logits[2],
                  "decode_step continuing the first branch, after the other one was run")  # fmt: skip
        self.step(first, start, self.logits[1], "decode_step(first token) a second time")


def tiny(ctx, name, index):
    key = f"tiny_{name}_{ctx.framework}"
    if key not in _cache:
        spec = ctx.fixture("tiny-configs.json")[name]
        cfg = RealConfig.from_json(spec["config"])
        weights = tiny_weights(spec["seed"], cfg.tied_embeddings, spec["embed_std"])
        total = float(sum(np.abs(w.astype(np.float64)).sum() for w in weights.values()))
        fixture = ctx.fixture("tiny.npz")
        check.that(
            abs(total - float(fixture[f"checksum_{name}"])) < 1e-6 * total,
            "test setup: the random weights differ from the ones the fixture was made with "
            "(numpy changed its generator?); rerun fixtures/make.py",
        )
        _cache[key] = ({k: ctx.fw.tensor(w) for k, w in weights.items()}, cfg, fixture)
    weights, cfg, fixture = _cache[key]
    return Model(ctx, weights, cfg, fixture, f"{name}_{index}")


def real(ctx, index):
    fixture = ctx.fixture("granite-4.0-350m.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    model = RealModel(ctx.asset(ASSET))
    key = f"weights_{ctx.framework}"
    if key not in _cache:
        # Bound to the framework once: each tensor is converted the first time it is used.
        _cache[key] = model.weights(ctx.fw)
    return Model(ctx, _cache[key], model.config, fixture, str(index), fixture["columns"])


# ---------------------------------------------------------------------------------
# Two three-layer random models: no download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: 8 greedy tokens after the one-token prompt [41]",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    model = tiny(ctx, "a", 2)
    check.that(model.tokens == WORKED_TOKENS, "test setup: the fixture is not the example")
    model.generate()


# Rejects rope_positions_restart, keys_rotated_again, caches_unrotated_keys,
# prefill_without_causal_mask, logits_of_first_row.
@case("prefill, then decode_step token by token: the logits of every step, 6-token prompt",
      public=True, assets=[])  # fmt: skip
def stepwise_logits(ctx):
    tiny(ctx, "a", 0).stepwise()
    tiny(ctx, "a", 1).stepwise()


@case("another configuration: other multipliers, RoPE base 10,000, an untied head",
      public=True, assets=[])  # fmt: skip
def other_configuration(ctx):
    tiny(ctx, "b", 0).stepwise()
    tiny(ctx, "b", 0).generate()


# Rejects mutates_cache_in_place.
@case("the cache is a value: two continuations of one cache, then the first one again",
      public=True, assets=[])  # fmt: skip
def cache_is_a_value(ctx):
    tiny(ctx, "b", 1).branch()
    tiny(ctx, "a", 1).branch()


# Rejects refeeds_last_prompt_token and includes_prompt.
@case("generate on both models and all prompts", assets=[])
def generate_all(ctx):
    for name in ("a", "b"):
        for index in (0, 1, 2):
            tiny(ctx, name, index).generate(label=f"generate(model {name}, prompt {index})")


@case("max_new_tokens = 1 and 2", assets=[])
def few_tokens(ctx):
    tiny(ctx, "b", 2).generate(1, label="generate(max_new_tokens=1)")
    tiny(ctx, "b", 2).generate(2, label="generate(max_new_tokens=2)")
    tiny(ctx, "b", 2).stepwise()


# ---------------------------------------------------------------------------------
# granite-4.0-350m: 28 layers, a cache of 4 KV heads per layer.
# ---------------------------------------------------------------------------------


@case("Granite 4.0 350M: 'A B C D E F G H I J K L' continues ' M N O P Q R S T U V W X'",
      public=True)  # fmt: skip
def real_alphabet(ctx):
    got = real(ctx, 0).generate()
    ctx.metric("generated_tokens", len(got), "tokens")


@case("Granite 4.0 350M: a line of Python, the logits of every step")
def real_code(ctx):
    real(ctx, 1).stepwise()


@case("Granite 4.0 350M: two continuations of one cache")
def real_branch(ctx):
    real(ctx, 1).branch()


@case("Granite 4.0 350M: a prompt of 768 tokens, then five decode steps", time_limit_s=60)
def real_long_prompt(ctx):
    real(ctx, 2).stepwise()


# Rejects no_cache: a decode_step that reads the 768 prompt tokens again is about eight
# times slower than one that reads its cache.
@case("Granite 4.0 350M: four decode steps after 768 tokens, within 3x of the reference",
      time_limit_s=120)  # fmt: skip
def real_decode_speed(ctx):
    model = real(ctx, 2)
    prompt = ctx.fw.tensor(model.prompt)
    tokens = model.tokens[:4]

    def decode_from_prompt(module):
        logits, cache = module.prefill(model.weights, model.cfg, prompt)
        ctx.fw.sync(logits, cache)  # the prompt is read before the clock starts

        def run():
            state, out = cache, None
            for token in tokens:
                out, state = module.decode_step(model.weights, model.cfg, token, state)
            return out, state  # forced by the judge before the clock stops

        return run

    ctx.compare_speed(
        decode_from_prompt(ctx.solution),
        decode_from_prompt(ctx.reference),
        max_ratio=MAX_RATIO,
        label="4 decode steps after a 768-token prompt",
        rounds=3,
        warmup=1,
        tokens=len(tokens),
    )
