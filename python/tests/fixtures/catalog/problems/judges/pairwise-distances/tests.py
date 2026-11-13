from whetstone.judge import case


@case("matches the reference", public=True)
def correct(ctx):
    ctx.check("pairwise_sq_dists", ctx.randn(6, 4), ctx.randn(5, 4), atol=1e-4)


@case("within 5x of the vectorized reference")
def fast_enough(ctx):
    ctx.check_speed("pairwise_sq_dists", ctx.randn(48, 16), ctx.randn(48, 16), max_ratio=5.0)
