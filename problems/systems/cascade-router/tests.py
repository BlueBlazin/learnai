"""Tests for cascade-router.

Every case builds scripted models, runs the solution, and compares the result, the
calls each model received, and the spend with what the statement's rule gives. The
rule is restated here in `expected` (so the expected values do not come from the
reference), and the worked example is compared with literal values.
"""

from whetstone.judge import case, check
from whetstone.kit import Counters, ScriptedChatModel, reply

KEYS = ["answer", "model", "confidence", "cost", "accepted"]


def make_models(specs):
    """specs: [(name, cost, {prompt: (content, logprobs)})] -> ScriptedChatModel list."""
    return [
        ScriptedChatModel(
            {p: reply(content, logprobs=lps) for p, (content, lps) in script.items()},
            name=name,
            cost=cost,
        )
        for name, cost, script in specs
    ]


def expected(prompt, specs, thresholds, budget):
    """(result, names of the models called) by the statement's rule."""
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}
    called = []
    for i, (name, cost, script) in enumerate(specs):
        if result["cost"] + cost > budget:
            break
        content, lps = script[prompt]
        called.append(name)
        result.update(answer=content, model=name, confidence=sum(lps) / len(lps))
        result["cost"] += cost
        if i == len(specs) - 1 or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break
    return result, called


def describe(prompt, specs, thresholds, budget):
    lines = [f"cascade({prompt!r}, models, thresholds={thresholds}, budget={budget})"]
    for name, cost, script in specs:
        content, lps = script[prompt]
        lines.append(f"  {name}: cost {cost}, replies {content!r} with logprobs {lps}")
    return "\n".join(lines) + "\n"


def run(ctx, prompt, specs, thresholds, budget, totals=None):
    """Run the solution on one prompt and check the result and what it cost."""
    models = make_models(specs)
    want, want_called = expected(prompt, specs, thresholds, budget)
    setup = describe(prompt, specs, thresholds, budget)
    got = ctx.call("cascade", prompt, models, list(thresholds), budget)

    check.that(isinstance(got, dict), "cascade: must return a dict", f"got: {type(got).__name__}")
    check.equal(sorted(got), sorted(KEYS), label="keys of the result")
    detail = f"{setup}expected: {want}\ngot:      {got}\n"

    # What was bought comes first: a wrong call explains a wrong answer.
    called = [m.name for m in models if m.counters.calls]
    for model in models:
        check.that(
            model.counters.calls <= 1,
            f"model {model.name!r} was called {model.counters.calls} times for one prompt",
            detail,
        )
    check.that(
        called == want_called,
        f"the models called were {called}, which is not what the cascade prescribes here",
        f"{setup}expected calls: {want_called}\n",
    )
    for model in models:
        for messages in model.calls:
            check.that(
                messages == [{"role": "user", "content": prompt}],
                f"model {model.name!r} did not receive the single user message with the prompt",
                f"it received: {messages}\n",
            )

    check.that(
        got["answer"] == want["answer"] and got["model"] == want["model"],
        f"the answer returned is {got['answer']!r} from {got['model']!r}, which is not the "
        "answer the cascade ends with",
        detail,
    )
    check.that(
        type(got["accepted"]) is bool and got["accepted"] == want["accepted"],
        f"accepted is {got['accepted']!r}, which is wrong for this run",
        detail,
    )
    if want["confidence"] is None:
        check.that(got["confidence"] is None, "confidence must be None when no model is called")
    else:
        check.that(
            isinstance(got["confidence"], float),
            f"confidence is a {type(got['confidence']).__name__}, not a float",
            detail,
        )
        check.allclose(got["confidence"], want["confidence"], rtol=0, atol=1e-9, label="confidence")
    check.that(
        got["cost"] == want["cost"],
        f"cost is {got['cost']!r}, which is not the sum of the costs of the models that "
        "should have been called",
        detail,
    )
    spent = sum(m.spent for m in models)
    check.that(
        spent == got["cost"],
        f"cost is {got['cost']!r}, but the models were paid {spent}",
        detail,
    )
    if totals is not None:
        totals.add("model_calls", len(called))
        totals.add("spend", spent)
    return got


FRANCE = "Capital of France?"
EXAMPLE = [
    ("small", 1, {FRANCE: ("Lyon", [-0.1, -0.5, -0.6])}),
    ("medium", 5, {FRANCE: ("Paris", [-0.25, -0.75])}),
    ("large", 20, {FRANCE: ("Paris", [-0.01, -0.03])}),
]


def one(prompt, *models):
    """Specs for a single prompt: one(prompt, (name, cost, content, logprobs), ...)."""
    return [(name, cost, {prompt: (content, lps)}) for name, cost, content, lps in models]


# Rejects sum_logprobs (-1.2 escalates past medium's -1.0), strict_threshold (medium's
# confidence equals its threshold), and calls_every_model (large is paid for).
@case("the worked example: small is rejected, medium accepted at its threshold", public=True)
def worked_example(ctx):
    got = run(ctx, FRANCE, EXAMPLE, [-0.2, -0.5], 30)
    check.equal(got["answer"], "Paris", label="answer")
    check.equal(got["model"], "medium", label="model")
    check.equal(got["cost"], 6, label="cost")
    check.equal(got["accepted"], True, label="accepted")


# Rejects budget_after_call (medium is called and paid for).
@case("the worked example with budget 5: the second call is not made", public=True)
def worked_example_budget(ctx):
    got = run(ctx, FRANCE, EXAMPLE, [-0.2, -0.5], 5)
    check.equal(got["answer"], "Lyon", label="answer")
    check.equal(got["cost"], 1, label="cost")
    check.equal(got["accepted"], False, label="accepted")


# Rejects sum_logprobs: twelve tokens at -0.05 have mean -0.05 and sum -0.6.
@case("a long confident answer is accepted, a one-token guess is escalated", public=True)
def length_does_not_matter(ctx):
    q = "Summarize the report."
    specs = one(q, ("small", 2, "a long summary", [-0.05] * 12), ("large", 30, "another", [-0.2]))
    run(ctx, q, specs, [-0.3], 100)
    q = "Yes or no?"
    specs = one(q, ("small", 2, "no", [-0.9]), ("large", 30, "yes", [-2.5]))
    run(ctx, q, specs, [-0.5], 100)


# Rejects budget_strict (a call that lands exactly on the budget is allowed).
@case("the budget may be spent exactly; a budget below the first price buys nothing",
      public=True)  # fmt: skip
def budget_edges(ctx):
    run(ctx, FRANCE, EXAMPLE, [-0.2, -0.5], 6)
    run(ctx, FRANCE, EXAMPLE, [-0.2, -0.3], 26)
    run(ctx, FRANCE, EXAMPLE, [-0.2, -0.3], 25)
    run(ctx, FRANCE, EXAMPLE, [-0.2, -0.5], 1)
    got = run(ctx, FRANCE, EXAMPLE, [-0.2, -0.5], 0)
    check.equal(got["answer"], None, label="answer when no model is affordable")


def workload(rng, n):
    """n prompts for three models; logprobs are multiples of 1/64, thresholds of 1/8."""
    costs = [("small", 1), ("medium", 6), ("large", 40)]
    scripts = [{} for _ in costs]
    prompts = []
    for j in range(n):
        prompt = f"question {j}"
        prompts.append(prompt)
        for level, script in enumerate(scripts):
            tokens = int(rng.integers(1, 13))
            scale = (40, 24, 8)[level]
            lps = [-int(k) / 64 for k in rng.integers(0, scale + 1, size=tokens)]
            script[prompt] = (f"{costs[level][0]} answer {j}", lps)
    specs = [(name, cost, script) for (name, cost), script in zip(costs, scripts, strict=True)]
    return prompts, specs


# Rejects calls_every_model and budget_after_call on the total spend.
@case("40 prompts through three models: calls and spend are exactly the cascade's", public=True)
def workload_cost(ctx):
    prompts, specs = workload(ctx.rng, 40)
    thresholds, budget = [-0.25, -0.125], 50
    totals = Counters("model_calls", "spend")
    want_calls = want_spend = 0
    for prompt in prompts:
        want, called = expected(prompt, specs, thresholds, budget)
        want_calls += len(called)
        want_spend += want["cost"]
        run(ctx, prompt, specs, thresholds, budget, totals)
    ctx.metric("always_large_spend", 40 * len(prompts), unit="units")
    check.cost(totals, "model_calls", exactly=want_calls, unit="calls")
    check.cost(totals, "spend", exactly=want_spend, unit="units")


# Rejects strict_threshold and budget_strict.
@case("a confidence exactly at its threshold is accepted")
def at_threshold(ctx):
    q = "q"
    specs = one(q, ("a", 1, "x", [-0.5, -0.25, -0.75]), ("b", 4, "y", [-1.0]))
    run(ctx, q, specs, [-0.5], 10)
    specs = one(q, ("a", 1, "x", [-0.125]), ("b", 4, "y", [-0.5, -1.5]), ("c", 9, "z", [-3.0]))
    run(ctx, q, specs, [-0.0625, -1.0], 14)
    run(ctx, q, specs, [-0.0625, -1.0], 5)


# Rejects threshold_on_last.
@case("the last model's answer is accepted however unsure it is")
def last_model_stands(ctx):
    q = "q"
    specs = one(q, ("a", 1, "x", [-2.0]), ("b", 4, "y", [-3.0, -4.0]), ("c", 9, "z", [-6.0]))
    run(ctx, q, specs, [-0.5, -0.25], 100)
    run(ctx, q, one(q, ("only", 3, "x", [-7.5, -0.5])), [], 3)


# Rejects most_confident_on_stop: the earlier reply is the more confident one.
@case("stopped by the budget: the answer is the last model's, not the most confident")
def stopped_by_budget(ctx):
    q = "q"
    specs = one(
        q,
        ("a", 2, "from a", [-0.25, -0.5]),
        ("b", 5, "from b", [-1.0, -0.5, -0.75]),
        ("c", 30, "from c", [-0.01]),
    )
    run(ctx, q, specs, [-0.125, -0.5], 20)
    run(ctx, q, specs, [-0.125, -0.5], 36)
    run(ctx, q, specs, [-0.125, -0.5], 37)


# Rejects mean_probability: [-0.01, -3.0] has mean -1.505, and log-mean-exp -0.65.
@case("one confident token does not hide an uncertain one")
def geometric_mean(ctx):
    q = "q"
    specs = one(q, ("a", 1, "x", [-0.01, -3.0]), ("b", 8, "y", [-0.3]))
    run(ctx, q, specs, [-1.0], 100)
    specs = one(q, ("a", 1, "x", [-0.02, -0.02, -0.02, -4.0]), ("b", 8, "y", [-0.3]))
    run(ctx, q, specs, [-0.9], 100)


@case("free models, equal prices, and a single model")
def degenerate_lists(ctx):
    q = "q"
    specs = one(q, ("a", 0, "x", [-1.0]), ("b", 0, "y", [-1.0]), ("c", 3, "z", [-1.0]))
    run(ctx, q, specs, [-0.5, -0.5], 0)
    run(ctx, q, specs, [-0.5, -1.0], 0)
    run(ctx, q, specs, [-0.5, -0.5], 3)
    run(ctx, q, one(q, ("only", 5, "x", [-0.5])), [], 4)


@case("random cascades of 1 to 6 models")
def random_cascades(ctx):
    rng = ctx.rng
    for trial in range(60):
        n = int(rng.integers(1, 7))
        costs = sorted(int(c) for c in rng.integers(0, 30, size=n))
        q = f"prompt {trial}"
        models = []
        for i, cost in enumerate(costs):
            tokens = int(rng.integers(1, 12))
            lps = [-int(k) / 64 for k in rng.integers(0, 96, size=tokens)]
            models.append((f"m{i}", cost, f"answer {i}", lps))
        thresholds = [-int(k) / 8 for k in rng.integers(0, 10, size=n - 1)]
        budget = int(rng.integers(0, sum(costs) + 3))
        run(ctx, q, one(q, *models), thresholds, budget)
