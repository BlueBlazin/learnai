from whetstone.judge import case, check


@case("sums a short list", public=True)
def short(ctx):
    check.equal(ctx.call("total", [1, 2, 3]), 6, label="total")


@case("sums an empty list", public=True)
def empty(ctx):
    check.equal(ctx.call("total", []), 0, label="total")


@case("negative numbers")
def negatives(ctx):
    xs = [-5, 2, -7]
    check.equal(ctx.call("total", xs), ctx.expected("total", xs), label="total")


@case("a long list")
def long_list(ctx):
    xs = [int(v) for v in ctx.rng.integers(0, 100, size=50)]
    check.equal(ctx.call("total", xs), ctx.expected("total", xs), label="total")
