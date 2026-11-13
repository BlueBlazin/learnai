import numpy as np

from whetstone.judge import case, check

PROBS = [0.5, 0.25, 0.125, 0.125]


@case("inverse CDF on hand-picked uniforms", public=True)
def hand_picked(ctx):
    for u, expected in [(0.0, 0), (0.49, 0), (0.5, 1), (0.74, 1), (0.8, 2), (0.99, 3)]:
        check.equal(ctx.call("sample", PROBS, u), expected, label=f"sample(probs, {u})")


@case("the samples follow the distribution")
def follows_distribution(ctx):
    probs = ctx.rng.dirichlet(np.ones(6)).tolist()
    uniforms = ctx.rng.random(20_000)  # randomness is an input: the test draws it
    samples = [ctx.call("sample", probs, float(u)) for u in uniforms]
    check.distribution(samples, probs, label="sample")
