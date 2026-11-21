"""Tests for block-diffusion-decode.

Two oracles, neither of them this problem's reference:

- `walk`, the loop of the statement in float64 numpy. It also draws the uniforms of a
  case while it runs, so that no decision of the correct loop sits on a float32 rounding
  boundary (the statement's Constraints): every `u_sample` is put inside a token's
  probability interval, and a draw is rejected (`Unsafe`) when a bound, a cut, an argmax,
  or the confidence test is closer than the margin. Given `wrong=`, it runs one named
  mistake instead; cases use that to assert that the rule they are written for changes
  the tokens or the number of steps on their fixture (an author's check, "internal:").
- `fixtures/shipped.json`: the output of `DiffusionGemmaGenerationMixin.generate` of
  `transformers` driven by the same toy denoiser and the same uniforms
  (`fixtures/make.py`), with the sizes of its encoder calls and its number of decoder
  calls.

The model is handed to the solution behind a guard (`Guarded`, a local helper as in
`indexer-cache-decode`): the solution may use `encode`, `denoise`, `vocab_size`,
`eos_id`, and `pad_id`; every other attribute, every attribute set, a copy, and a cache
the object did not return itself are refused and recorded. Work is also logged by the
test's own subclass (`Denoiser.work`), which the kit's counters must agree with.

Shapes are few, because JAX compiles once per shape: canvases of 8 tokens over the
toy's vocabulary of 32, the worked example's 4 tokens over 8, and one scenario with
canvases of a single token over 4.
"""

import sys
from typing import ClassVar

import numpy as np

from learnai.judge import case, check
from learnai.judge.errors import CheckFailure
from learnai.kit import BlockDenoiser

FN = "diffusion_generate"
S1, S2, S3 = "1: one canvas", "2: canvases over a cached prefix", "3: cost"
MARGIN = 1e-3  # the distance from every rounding boundary, as stated in Constraints
PROMPT = [5, 9, 2, 7]
# The generation defaults of DiffusionGemma, except canvas_length (256 there) and
# max_steps (48 there): `_get_default_generation_params` in transformers 5.23.
SHIPPED = {
    "canvas_length": 8,
    "max_canvases": 3,
    "max_steps": 12,
    "t_min": 0.4,
    "t_max": 0.8,
    "entropy_bound": 0.1,
    "stability": 1,
    "confidence": 0.005,
}
# Looser thresholds, under which a canvas stops while positions are still unsettled.
LOOSE = {**SHIPPED, "t_min": 0.2, "t_max": 1.0, "entropy_bound": 1.0, "stability": 2,
         "confidence": 0.5}  # fmt: skip


# -- the guard ------------------------------------------------------------------------


class Guarded:
    """Mixin for a kit object handed to the solution (a local helper, not in the kit).

    Code in the solution file may use the names in `allowed` and nothing else: every
    other attribute (the counters, `with_framework`, the settings a copy could be built
    from, private attributes, `__dict__`) and every attribute set is refused. The access
    raises a failure and is recorded, so catching the exception does not help. Kit and
    test code are not restricted. It is a guard, not a sandbox: `object.__getattribute__`
    goes around it, as the editorial says.
    """

    allowed: ClassVar[frozenset] = frozenset()
    sealed: ClassVar[frozenset] = frozenset()  # readable by nobody unless `_open` is set
    label = "the kit object"
    solution_file = None  # set before the solution is called
    violations: ClassVar[list] = []  # shared by every guarded object

    def _refuse(self, name, verb):
        if sys._getframe(2).f_code.co_filename != Guarded.solution_file:
            return
        label = object.__getattribute__(self, "label")
        message = f"`{label}.{name}` may not be {verb} in this problem"
        Guarded.violations.append(message)
        raise CheckFailure(message)

    def __getattribute__(self, name):
        if name not in type(self).allowed:
            object.__getattribute__(self, "_refuse")(name, "used")
        if name in type(self).sealed and not object.__getattribute__(self, "_open"):
            label = object.__getattribute__(self, "label")
            message = f"`{label}.{name}` may not be read outside the model's own methods"
            Guarded.violations.append(message)
            raise CheckFailure(message)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_refuse")(name, "set")
        object.__setattr__(self, name, value)


def violation(message):
    Guarded.violations.append(message)
    raise CheckFailure(message)


class Denoiser(Guarded, BlockDenoiser):
    """The model the solution gets: guarded, and with a log of the work it did.

    `work` holds ("encode", tokens, cached before) and ("denoise", canvas length, cached,
    self-conditioned) in call order. The weights can be read only while one of the
    methods below is running, so the kit's arithmetic cannot be borrowed as an unbound
    function (`BlockDenoiser._denoise(model, ...)`) without it showing in the log. A
    cache is accepted only if this very object returned it, so nothing computed on
    another instance can be mixed in.
    """

    label = "model"
    allowed: ClassVar[frozenset] = frozenset(
        {"encode", "denoise", "vocab_size", "eos_id", "pad_id"}
    )
    sealed: ClassVar[frozenset] = frozenset({"_w"})

    def __init__(self, *args, **kwargs):
        self._open = False
        BlockDenoiser.__init__(self, *args, **kwargs)
        self.work = []
        self._made = []  # the caches this object returned

    def _own(self, cache):
        if not any(cache is made for made in self._made):
            violation("`model` was given a cache that it did not return itself")

    def encode(self, tokens, cache=None):
        if cache is not None:
            self._own(cache)
        self._open = True
        try:
            longer = BlockDenoiser.encode(self, tokens, cache)
        finally:
            self._open = False
        before = 0 if cache is None else cache.length
        self.work.append(("encode", longer.length - before, before))
        self._made.append(longer)
        return longer

    def denoise(self, canvas, cache, self_cond_logits=None):
        self._own(cache)
        return BlockDenoiser.denoise(self, canvas, cache, self_cond_logits)

    def _denoise(self, ids, cache, previous):
        self.work.append(("denoise", len(ids), cache.length, previous is not None))
        self._open = True
        try:
            return BlockDenoiser._denoise(self, ids, cache, previous)
        finally:
            self._open = False

    def _no_copy(self, *args):
        violation("`model` may not be copied in this problem")

    __copy__ = __deepcopy__ = __reduce_ex__ = _no_copy


# -- the loop of the statement, in float64 --------------------------------------------


class Unsafe(Exception):
    """A decision of the correct loop is closer to a rounding boundary than MARGIN."""


def probabilities(scaled):
    z = scaled.astype(np.float64)
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def noise_uniforms(rng, vocab, *shape):
    """Uniforms whose floor(u * vocab) is clear of rounding: the middle of a token's bin."""
    return ((rng.integers(0, vocab, shape) + rng.uniform(0.25, 0.75, shape)) / vocab).astype(
        np.float32
    )


def walk(model, prompt, cfg, uniforms=None, rng=None, wrong=None, strict=None):
    """Run the loop. Returns {"tokens", "steps" (per canvas), "uniforms", "flags", "log"}.

    With `rng`, the uniforms are drawn here, `u_sample` step by step from the
    probabilities the step sees. With `uniforms`, they are used as given. When `strict`
    (the default when drawing), `Unsafe` is raised if a decision is within MARGIN of
    flipping. `wrong` names one mistake to make. `flags` has one string per canvas with
    two characters per step: "s" if the step was stable, "c" if it was confident. `log`
    has one dict per step, for the worked example.
    """
    length, canvases, steps = cfg["canvas_length"], cfg["max_canvases"], cfg["max_steps"]
    vocab, drawing = model.vocab_size, uniforms is None
    if drawing:
        u_init = noise_uniforms(rng, vocab, canvases, length)
        u_noise = noise_uniforms(rng, vocab, canvases, steps, length)
        u_sample = rng.uniform(0.02, 0.98, (canvases, steps, length)).astype(np.float32)
    else:
        u_init, u_sample, u_noise = uniforms
        if wrong == "first_canvas_uniforms":
            u_init, u_sample, u_noise = (np.broadcast_to(u[:1], u.shape) for u in uniforms)

    strict = drawing if strict is None else strict

    def near(distance, margin=MARGIN):
        return strict and distance < margin

    def binned(u):  # floor(u * vocab), which must not sit on an integer
        product = u.astype(np.float64) * vocab
        if near(np.abs(product - np.round(product)).min()):
            raise Unsafe("a noise uniform on a token boundary")
        return np.floor(u * np.float32(vocab)).astype(np.int64)

    cache = model.encode(np.asarray(prompt))
    out, taken, flags, log, previous = list(prompt), [], [], [], None
    for c in range(canvases):
        canvas = binned(u_init[c])
        if wrong != "self_cond_across_canvases":
            previous = None
        marks = ""
        if c == 0 or wrong != "drafts_across_canvases":
            drafts = []
        for j in range(steps):
            remaining = steps - j - (wrong == "temperature_off_by_one")
            temperature = cfg["t_min"] + (cfg["t_max"] - cfg["t_min"]) * remaining / steps
            given = None if wrong == "no_self_conditioning" else previous
            raw = np.asarray(model.fw.numpy(model.denoise(model.fw.tensor(canvas), cache, given)))
            scaled = raw / np.float32(temperature)
            p = probabilities(scaled)
            cdf = np.cumsum(p, axis=-1)
            if drawing:  # a token per position, and a uniform in the middle of its interval
                low = np.concatenate((np.zeros((length, 1)), cdf[:, :-1]), axis=-1)
                for i in range(length):
                    token = min(int(np.searchsorted(cdf[i], rng.random(), side="right")), vocab - 1)
                    if p[i, token] < 4 * MARGIN:
                        token = int(p[i].argmax())
                    u_sample[c, j, i] = low[i, token] + p[i, token] * rng.uniform(0.3, 0.7)
            threshold = u_sample[c, j].astype(np.float64)[:, None] * cdf[:, -1:]
            sampled = (cdf <= threshold).sum(axis=-1)
            if near(np.abs(cdf - threshold).min()):
                raise Unsafe("u_sample on a cumulative sum")

            h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)
            order = np.argsort(h, kind="stable")
            before = np.cumsum(h[order]) - h[order]
            keep = before <= cfg["entropy_bound"]
            off = np.abs(before - cfg["entropy_bound"])[before > 0]
            if off.size and near(off.min()):
                raise Unsafe("the entropy bound on a sum")
            count = int(keep.sum())
            if count < length and near(h[order[count]] - h[order[count - 1]]):
                raise Unsafe("equal entropies at the cut")
            accepted = np.zeros(length, dtype=bool)
            accepted[order] = keep
            start = canvas
            canvas = np.where(accepted, sampled, binned(u_noise[c, j]))

            draft = scaled.argmax(axis=-1)
            compared = canvas if wrong == "stability_on_sampled_canvas" else draft
            top = np.sort(scaled.astype(np.float64), axis=-1)[:, -2:]
            if near((top[:, 1] - top[:, 0]).min()):
                raise Unsafe("two largest logits too close")
            need = cfg["stability"]
            if wrong == "stability_last_draft_only":
                recent = drafts[-1:] if need else []
            elif wrong == "zero_stability_slices_all":
                recent = drafts[-need:]
            else:
                recent = drafts[len(drafts) - need :]
            stable = len(drafts) >= need and all(np.array_equal(compared, old) for old in recent)
            mean = float(h.sum() if wrong == "confidence_sum_not_mean" else h.mean())
            if near(abs(mean - cfg["confidence"]), min(MARGIN, 0.1 * cfg["confidence"])):
                raise Unsafe("mean entropy on the confidence threshold")
            confident = mean < cfg["confidence"]
            marks += ("s" if stable else ".") + ("c" if confident else ".")
            log.append({"canvas": c, "step": j, "temperature": temperature, "input": start,
                        "draft": draft, "accepted": accepted, "output": canvas,
                        "mean_entropy": mean, "stable": stable, "confident": confident})  # fmt: skip
            drafts.append(canvas if wrong == "stability_on_sampled_canvas" else draft)
            previous = model.fw.tensor(raw if wrong == "self_cond_raw_logits" else scaled)
            stop = {
                "stops_on_stability_alone": stable,
                "stops_on_confidence_alone": confident,
                "no_early_stop": False,
            }.get(wrong, stable and confident)
            if stop:
                break
        taken.append(len(marks) // 2)
        flags.append(marks)
        final = canvas.copy() if wrong == "appends_sampled_canvas" else draft.copy()
        is_eos = final == model.eos_id
        if is_eos.any() and wrong != "ignores_eos":
            if wrong != "eos_not_cut":
                final[int(np.argmax(is_eos)) + 1 :] = model.pad_id
            out += final.tolist()
            break
        out += final.tolist()
        if c + 1 < canvases:
            kept = canvas if wrong == "encodes_noisy_canvas" else final
            cache = model.encode(kept, cache)
    return {
        "tokens": out,
        "steps": taken,
        "uniforms": (u_init, u_sample, u_noise),
        "flags": flags,
        "log": log,
    }


# -- running a case -------------------------------------------------------------------


class Scenario:
    """A model, a prompt, a configuration, safe uniforms, and what the loop does with them."""

    def __init__(self, ctx, *, seed, sharpness, eos_bias=-100.0, prompt=PROMPT, vocab=32,
                 base=SHIPPED, uniforms=None, eos_id=1, pad_id=0, **overrides):  # fmt: skip
        self.ctx, self.prompt, self.cfg = ctx, list(prompt), {**base, **overrides}

        def maker(cls=BlockDenoiser):
            model = cls(ctx.fw, seed=seed, sharpness=sharpness, eos_bias=eos_bias,
                        vocab_size=vocab)  # fmt: skip
            # The kit reads both ids from the object, so a model with other special
            # tokens is the same model with two attributes set.
            model.eos_id, model.pad_id = eos_id, pad_id
            return model

        self.maker = maker
        if uniforms is not None:  # given (the worked example): they must be safe as they are
            given = tuple(np.asarray(u, dtype=np.float32) for u in uniforms)
            self.want = walk(self.maker(), self.prompt, self.cfg, uniforms=given, strict=True)
        else:
            for draw in range(50):  # the first draw whose decisions are all clear of rounding
                try:
                    rng = np.random.default_rng([seed, draw])
                    self.want = walk(self.maker(), self.prompt, self.cfg, rng=rng)
                    break
                except Unsafe:
                    continue
            else:
                raise RuntimeError("authoring error: no safe uniforms for this scenario")
        self.uniforms = self.want["uniforms"]
        self.model = None

    def mistake(self, wrong):
        """What the loop with one mistake does with the same uniforms."""
        return walk(self.maker(), self.prompt, self.cfg, uniforms=self.uniforms, wrong=wrong)

    def must_change(self, *mistakes, what="tokens"):
        """Author's check: each mistake changes the tokens (or the steps) on this fixture."""
        for wrong in mistakes:
            if self.mistake(wrong)[what] == self.want[what]:
                raise RuntimeError(f"internal: {wrong} does not change the {what} of this case")

    def run(self, expected=None):
        """Call the solution on a fresh guarded model and compare the tokens."""
        ctx = self.ctx
        self.model = self.maker(Denoiser)
        Guarded.solution_file = ctx.solution.__file__
        Guarded.violations.clear()
        tokens = ctx.call(FN, self.model, list(self.prompt), dict(self.cfg), self.uniforms)
        check.that(not Guarded.violations, Guarded.violations[0] if Guarded.violations else "")
        check.that(
            isinstance(tokens, (list, tuple, np.ndarray)),
            f"{FN}: the result is not a list of tokens",
        )
        tokens = [int(token) for token in np.asarray(tokens).reshape(-1)]
        check.equal(tokens, self.want["tokens"] if expected is None else expected, label="tokens")
        return tokens

    def cost(self, public=False):
        """Stage 3: the tokens first, then the work the model was asked to do.

        A public case reports the counters with their limits. A hidden case says which
        counter is off and by nothing more: its message must not give the expected count.
        """
        self.run()
        prompt, length = len(self.prompt), self.cfg["canvas_length"]
        canvases, steps = len(self.want["steps"]), sum(self.want["steps"])
        counters = self.model.counters
        expected = (
            ("encoder_tokens", prompt + (canvases - 1) * length, "tokens"),
            ("encoder_calls", canvases, "calls"),
            ("denoise_calls", steps, "calls"),
        )
        for name, exactly, unit in expected:
            if public:
                check.cost(counters, name, exactly=exactly, unit=unit)
            else:
                got = getattr(counters, name)
                check.that(
                    got == exactly,
                    f"{name}: {got} {unit} is not the number the statement's cost table gives",
                )
        # The test's own log must tell the same story as the kit's counters.
        encoded = [entry[1] for entry in self.model.work if entry[0] == "encode"]
        denoised = [entry for entry in self.model.work if entry[0] == "denoise"]
        check.that(
            sum(encoded) == counters.encoder_tokens and len(denoised) == counters.denoise_calls,
            "the model did work that its counters do not show",
        )


# Every case is one of these scenarios (keyword arguments of Scenario), or a sweep.
# fixtures/make.py runs the library's loop on each of them.
EXAMPLE_UNIFORMS = (
    [[0.97, 0.44, 0.2, 0.69], [0.81, 0.29, 0.3, 0.08]],
    [
        [[0.4, 0.58, 0.58, 0.34], [0.43, 0.57, 0.32, 0.33], [0.92, 0.92, 0.92, 0.71]],
        [[0.78, 0.41, 0.94, 0.74], [0.26, 0.7, 0.43, 0.35], [0.21, 0.45, 0.51, 0.42]],
    ],
    [
        [[0.71, 0.91, 0.68, 0.94], [0.69, 0.94, 0.43, 0.09], [0.56, 0.71, 0.18, 0.7]],
        [[0.83, 0.66, 0.17, 0.7], [0.54, 0.16, 0.67, 0.68], [0.83, 0.83, 0.79, 0.05]],
    ],
)
ONE = {"max_canvases": 1}
# fmt: off
SCENARIOS = {
    "example": {"seed": 18, "sharpness": 3.0, "eos_bias": 1.0, "vocab": 8, "prompt": [3, 6, 2],
                "canvas_length": 4, "max_canvases": 2, "max_steps": 3,
                "uniforms": EXAMPLE_UNIFORMS},
    # one canvas
    "example_first": {"seed": 18, "sharpness": 3.0, "eos_bias": 1.0, "vocab": 8,
                      "prompt": [3, 6, 2], "canvas_length": 4, "max_steps": 3, **ONE,
                      "uniforms": tuple(u[:1] for u in EXAMPLE_UNIFORMS)},
    "early": {"seed": 2, "sharpness": 3.0, **ONE},
    "limit": {"seed": 4, "sharpness": 1.0, **ONE},
    "repeats": {"seed": 6, "sharpness": 1.0, **ONE},
    "never_both": {"seed": 2, "sharpness": 1.0, "base": LOOSE, **ONE},
    "eos_inside": {"seed": 4, "sharpness": 3.0, "eos_bias": 3.0, **ONE},
    "eos_last": {"seed": 34, "sharpness": 3.0, "eos_bias": 2.5, **ONE},
    "eos_first": {"seed": 0, "sharpness": 3.0, "eos_bias": 100.0, **ONE},
    "zero": {"seed": 1, "sharpness": 3.0, "stability": 0, **ONE},
    "first_step": {"seed": 0, "sharpness": 1.5, "stability": 0, "confidence": 3.0, **ONE},
    "two": {"seed": 1, "sharpness": 6.0, "stability": 2, **ONE},
    # several canvases
    "three": {"seed": 1, "sharpness": 3.0},
    "three_limit": {"seed": 4, "sharpness": 1.0},
    "eos_second": {"seed": 2, "sharpness": 3.0, "eos_bias": 2.0},
    "loose": {"seed": 9, "sharpness": 1.5, "base": LOOSE},
    "loose_b": {"seed": 5, "sharpness": 1.5, "base": LOOSE},
    "loose_c": {"seed": 2, "sharpness": 1.5, "base": LOOSE},
    "loose_zero": {"seed": 0, "sharpness": 1.5, "base": LOOSE, "stability": 0},
    "eos_prompt": {"seed": 1, "sharpness": 3.0, "prompt": [5, 1, 2, 7]},
    "first_step_two": {"seed": 0, "sharpness": 1.5, "stability": 0, "confidence": 3.0,
                       "max_canvases": 2},
    "long_prompt": {"seed": 3, "sharpness": 3.0, "max_canvases": 4, "max_steps": 6,
                    "prompt": [9, 30, 4, 4, 17, 2, 8, 21, 6], "t_min": 0.25, "t_max": 1.5,
                    "entropy_bound": 0.5},
    "one_token_prompt": {"seed": 5, "sharpness": 3.0, "max_canvases": 2, "prompt": [14]},
    # other special tokens: the token 1 is ordinary text here
    "other_ids": {"seed": 28, "sharpness": 3.0, "eos_bias": 3.0, "eos_id": 28, "pad_id": 3},
    "other_ids_b": {"seed": 28, "sharpness": 3.0, "eos_bias": 2.0, "eos_id": 16, "pad_id": 5},
    "tiny_canvases": {"seed": 11, "sharpness": 1.5, "vocab": 4, "prompt": [3, 2],
                      "canvas_length": 1, "max_canvases": 5, "max_steps": 6, "confidence": 3.0},
}
# fmt: on


def scenario(ctx, name):
    """The scenario `name`, checked against the library's loop on the same inputs."""
    made = Scenario(ctx, **SCENARIOS[name])
    theirs = ctx.fixture("shipped.json")[name]
    length, canvases = made.cfg["canvas_length"], len(made.want["steps"])
    ours = {
        "tokens": made.want["tokens"],
        "encoder_calls": [len(made.prompt)] + [length] * (canvases - 1),
        "denoise_calls": sum(made.want["steps"]),
    }
    if ours != theirs:
        raise RuntimeError(f"internal: {name} disagrees with fixtures/shipped.json; run make.py")
    return made


# -- stage 1: one canvas --------------------------------------------------------------


@case("the worked example with max_canvases = 1: its first canvas", public=True, stage=S1)
def example_first_canvas(ctx):
    made = scenario(ctx, "example_first")
    assert made.want["steps"] == [2] and made.want["flags"] == ["..sc"]
    made.run(expected=[3, 6, 2, 4, 4, 4, 4])


@case("one canvas that stops early: the draft repeated and the entropy is low",
      public=True, stage=S1)  # fmt: skip
def stops_early(ctx):
    made = scenario(ctx, "early")
    assert made.want["steps"] == [4] and made.want["flags"][0].endswith("sc")
    made.must_change("no_self_conditioning", "self_cond_raw_logits")
    made.must_change("no_early_stop", what="steps")
    made.run()


# Rejects appends_sampled_canvas: after the last of 12 steps some positions of the
# canvas are noise or a sample that is not the most likely token.
@case("one canvas that runs out of steps: the text gets the last draft", public=True, stage=S1)
def step_limit(ctx):
    made = scenario(ctx, "limit")
    assert made.want["steps"] == [12] and "sc" not in made.want["flags"][0]
    made.must_change("appends_sampled_canvas", "temperature_off_by_one")
    made.run()


# Rejects stops_on_stability_alone: the draft repeats at eight steps of twelve.
@case("the draft repeats, but the mean entropy stays above the threshold",
      public=True, stage=S1)  # fmt: skip
def stable_not_confident(ctx):
    made = scenario(ctx, "repeats")
    assert made.want["flags"][0].count("s.") == 8 and "c" not in made.want["flags"][0]
    made.must_change("stops_on_stability_alone")
    made.run()


# Rejects both one-sided stopping rules: with stability 2 and looser thresholds, three
# steps are stable and two others confident, never the same step.
@case("stable at some steps, confident at others, never both: all 12 steps run",
      public=True, stage=S1)  # fmt: skip
def never_both(ctx):
    made = scenario(ctx, "never_both")
    flags = made.want["flags"][0]
    assert "s." in flags and ".c" in flags and "sc" not in flags
    made.must_change("stops_on_stability_alone", "stops_on_confidence_alone")
    made.run()


@case("an end-of-sequence token at position 5: the rest of the canvas is padding",
      public=True, stage=S1)  # fmt: skip
def eos_inside(ctx):
    made = scenario(ctx, "eos_inside")
    assert made.want["tokens"][len(PROMPT) :] == [26, 10, 27, 27, 27, 1, 0, 0]
    made.must_change("eos_not_cut")
    made.run()


@case("an end-of-sequence token at the last position, and one at the first", stage=S1)
def eos_at_the_ends(ctx):
    for name, position in (("eos_last", 7), ("eos_first", 0)):
        made = scenario(ctx, name)
        assert made.want["tokens"][len(PROMPT) :].index(1) == position
        made.run()


# With stability 0 a canvas stops at the first confident step, whatever the drafts were.
@case("stability 0", stage=S1)
def zero_stability(ctx):
    made = scenario(ctx, "zero")
    made.must_change("stops_on_stability_alone")
    made.must_change("zero_stability_slices_all", what="steps")
    made.run()
    made = scenario(ctx, "first_step")  # a threshold that the first step meets
    assert made.want["steps"] == [1]
    made.must_change("appends_sampled_canvas")
    made.must_change("no_early_stop", what="steps")
    made.run()


@case("stability 2", stage=S1)
def two_repeats(ctx):
    made = scenario(ctx, "two")
    assert made.want["flags"] == [".......csc"]
    made.run()


@case("other models and settings, one canvas", stage=S1)
def one_canvas_sweep(ctx):
    for seed, sharpness, overrides in (
        (11, 2.0, {}),
        (12, 3.0, {"max_steps": 6, "t_min": 0.25, "t_max": 1.5}),
        (13, 1.5, {"base": LOOSE, "stability": 1}),
        (14, 6.0, {"entropy_bound": 0.02}),
        (15, 1.5, {"base": LOOSE, "eos_bias": 1.5}),
    ):
        Scenario(ctx, seed=seed, sharpness=sharpness, **ONE, **overrides).run()


# -- stage 2: several canvases --------------------------------------------------------


@case("the worked example: two canvases of 4 tokens over a vocabulary of 8",
      public=True, stage=S2)  # fmt: skip
def worked_example(ctx):
    made = scenario(ctx, "example")
    assert made.want["steps"] == [2, 3] and made.want["flags"] == ["..sc", "..s..."]
    made.must_change(
        "appends_sampled_canvas", "stops_on_stability_alone", "self_cond_across_canvases",
        "temperature_off_by_one", "first_canvas_uniforms", "eos_not_cut",
    )  # fmt: skip
    made.run(expected=[3, 6, 2, 4, 4, 4, 4, 4, 1, 0, 0])


@case("three canvases, each stopping early", public=True, stage=S2)
def three_canvases(ctx):
    made = scenario(ctx, "three")
    assert made.want["steps"] == [6, 4, 5]
    made.must_change("self_cond_across_canvases", "first_canvas_uniforms", "no_self_conditioning")
    made.run()


# Rejects encodes_noisy_canvas: when a canvas ends at the step limit, the canvas the
# sampler left behind is not the draft, and the next canvas is conditioned on the draft.
@case("three canvases that run out of steps: the cache gets the drafts", public=True, stage=S2)
def three_at_the_limit(ctx):
    made = scenario(ctx, "three_limit")
    assert made.want["steps"] == [12, 12, 12]
    made.must_change("encodes_noisy_canvas", "appends_sampled_canvas", "first_canvas_uniforms")
    made.run()


@case("an end-of-sequence token in the second canvas ends the text", public=True, stage=S2)
def eos_second_canvas(ctx):
    made = scenario(ctx, "eos_second")
    assert made.want["tokens"][-8:] == [7, 8, 8, 8, 1, 0, 0, 0]
    made.must_change("eos_not_cut", "ignores_eos")
    made.run()


# Rejects hardcoded_ids: the model's end-of-sequence token is 28 and its padding token
# 3, and the token 1 occurs three times in the first canvas without ending anything.
@case("a model whose end-of-sequence token is 28 and whose padding token is 3",
      public=True, stage=S2)  # fmt: skip
def other_special_tokens(ctx):
    made = scenario(ctx, "other_ids")
    tokens = made.want["tokens"][len(PROMPT) :]
    assert tokens == [10, 7, 7, 7, 1, 1, 1, 10, 10, 10, 10, 10, 7, 28, 3, 3]
    made.run()


# Looser thresholds stop a canvas while positions are unsettled, so nearly every rule
# decides the tokens here.
@case("looser thresholds and stability 2", public=True, stage=S2)
def loose_thresholds(ctx):
    made = scenario(ctx, "loose")
    made.must_change(
        "appends_sampled_canvas", "encodes_noisy_canvas", "no_self_conditioning",
        "self_cond_raw_logits", "stops_on_confidence_alone", "no_early_stop",
        "stability_last_draft_only", "temperature_off_by_one", "first_canvas_uniforms",
    )  # fmt: skip
    made.run()


# Rejects eos_anywhere: only a generated token ends the text.
@case("an end-of-sequence token in the prompt ends nothing", public=True, stage=S2)
def eos_in_the_prompt(ctx):
    made = scenario(ctx, "eos_prompt")
    assert len(made.want["steps"]) == 3 and 1 not in made.want["tokens"][len(made.prompt) :]
    made.run()


# Rejects zero_stability_slices_all at the level of tokens.
@case("stability 0: a canvas stops at its first confident step", public=True, stage=S2)
def zero_stability_canvases(ctx):
    made = scenario(ctx, "loose_zero")
    made.must_change("zero_stability_slices_all", "no_early_stop")
    made.run()
    made = scenario(ctx, "first_step_two")
    assert made.want["steps"] == [1, 1]
    made.must_change("encodes_noisy_canvas", "self_cond_across_canvases")
    made.run()


@case("nothing passed in is modified", public=True, stage=S2)
def inputs_unchanged(ctx):
    made = scenario(ctx, "three")
    model = made.maker(Denoiser)
    Guarded.solution_file = ctx.solution.__file__
    Guarded.violations.clear()
    prompt, cfg = list(made.prompt), dict(made.cfg)
    tensors = tuple(ctx.fw.tensor(u) for u in made.uniforms)
    tokens = ctx.fw.to_numpy(ctx.solution.diffusion_generate(model, prompt, cfg, tensors))
    check.that(not Guarded.violations, Guarded.violations[0] if Guarded.violations else "")
    check.equal([int(token) for token in tokens], made.want["tokens"], label="tokens")
    check.equal(prompt, made.prompt, label="prompt after the call")
    check.that(cfg == made.cfg, "cfg was changed by the call")
    for name, tensor, before in zip(("u_init", "u_sample", "u_noise"), tensors, made.uniforms,
                                    strict=True):  # fmt: skip
        check.allclose(ctx.fw.numpy(tensor), before, rtol=0, atol=0, label=f"{name} after the call")


@case("looser thresholds, other models", stage=S2)
def loose_others(ctx):
    for name in ("loose_b", "loose_c"):
        made = scenario(ctx, name)
        made.must_change("stops_on_confidence_alone", "stability_last_draft_only")
        made.run()


@case("other prompt lengths, canvas counts, and schedules", stage=S2)
def other_lengths(ctx):
    for name in ("long_prompt", "one_token_prompt", "other_ids_b"):
        scenario(ctx, name).run()
    for seed in (21, 22):
        Scenario(ctx, seed=seed, sharpness=2.0, max_canvases=2).run()


# -- stage 3: cost --------------------------------------------------------------------


@case("three canvases: the prompt is encoded once, and two canvases once each",
      public=True, stage=S3)  # fmt: skip
def cost_three_canvases(ctx):
    scenario(ctx, "three").cost(public=True)


# Rejects encodes_last_canvas and the mutants that go on after the end of the text.
@case("the text ends in the second canvas: one canvas is encoded", public=True, stage=S3)
def cost_eos(ctx):
    scenario(ctx, "eos_second").cost(public=True)


@case("three canvases at the step limit: 36 denoising calls", public=True, stage=S3)
def cost_step_limit(ctx):
    scenario(ctx, "three_limit").cost(public=True)


# Rejects the mutants that stop a step early with the same tokens.
@case("stability 2: the canvas stops at its fifth step, not its fourth", public=True, stage=S3)
def cost_two_repeats(ctx):
    made = scenario(ctx, "two")
    made.must_change("stops_on_confidence_alone", "stability_last_draft_only", what="steps")
    made.cost(public=True)


# Rejects drafts_across_canvases. The canvases are single tokens and the model repeats
# itself, so the first draft of a canvas often equals the last draft of the canvas
# before; it must not count as a repeat.
@case("canvases of one token: each takes two steps, though the text repeats",
      public=True, stage=S3)  # fmt: skip
def cost_tiny_canvases(ctx):
    made = scenario(ctx, "tiny_canvases")
    assert made.want["steps"] == [2, 2, 2, 2, 2]
    made.must_change("drafts_across_canvases", what="steps")
    made.cost(public=True)


@case("stability 0 and a single step per canvas", stage=S3)
def cost_zero_stability(ctx):
    made = scenario(ctx, "zero")
    made.must_change("zero_stability_slices_all", what="steps")
    made.cost()
    scenario(ctx, "first_step_two").cost()


@case("the worked example, looser thresholds, and other lengths", stage=S3)
def cost_others(ctx):
    for name in ("example_first", "example", "loose", "eos_prompt", "long_prompt",
                 "one_token_prompt", "early", "other_ids"):  # fmt: skip
        scenario(ctx, name).cost()
