from learnai.judge import case, check


@case("sums a short list", public=True)
def short(ctx):
    check.equal(ctx.call("total", [1, 2, 3]), 6, label="total")
