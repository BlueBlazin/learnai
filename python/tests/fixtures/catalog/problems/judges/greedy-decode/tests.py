from whetstone.judge import case, check
from whetstone.kit import TinyLM


@case("generates the same tokens as the reference", public=True)
def same_tokens(ctx):
    lm = TinyLM(ctx.fw, seed=3)
    prompt = [0, 17, 42, 5]
    got = ctx.call("greedy_decode", lm, prompt, 12)
    check.equal(got, ctx.expected("greedy_decode", lm, prompt, 12), label="greedy_decode")


@case("each token goes through the model once", public=True)
def linear_cost(ctx):
    lm = TinyLM(ctx.fw, seed=3)
    prompt = [0, 17, 42, 5]
    ctx.call("greedy_decode", lm, prompt, 12)
    # The prompt once, then every generated token except the last one.
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + 12 - 1, unit="tokens")
    check.cost(lm.counters, "forward_calls", at_most=12)


@case("a longer generation on another model")
def longer(ctx):
    lm = TinyLM(ctx.fw, seed=11)
    prompt = [0, 9]
    got = ctx.call("greedy_decode", lm, prompt, 40)
    check.equal(got, ctx.expected("greedy_decode", lm, prompt, 40), label="greedy_decode")
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + 40 - 1, unit="tokens")
