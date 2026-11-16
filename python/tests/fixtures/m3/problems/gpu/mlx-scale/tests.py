import numpy as np

from learnai.judge import case, check
from learnai.kit import mlx_memory


@case("scales a small array", public=True)
def small(ctx):
    ctx.check("scale", np.array([1.0, -2.0, 0.5], dtype=np.float32), 3.0)


@case("random arrays and factors")
def random_arrays(ctx):
    def prop(rng):
        ctx.check("scale", ctx.randn(4, 5), float(rng.uniform(-3, 3)))

    check.forall(prop, trials=5)


# Rejects promotes: the float32 result of a float16 input is twice the size.
@case("a float16 input stays float16 and stays small", requires="mlx-gpu")
def half_precision(ctx):
    x = ctx.fw.tensor(ctx.randn(1024, 1024), dtype=np.float16)
    ctx.fw.sync(x)
    with mlx_memory() as mem:
        y = ctx.solution.scale(x, 2.0)
        ctx.fw.sync(y)
    check.that(str(y.dtype).endswith("float16"), f"scale: got dtype {y.dtype} for a float16 input")
    check.memory(mem, at_most_mib=3.0)  # the float16 result is 2 MiB, a float32 one 4 MiB


@case("as fast as the reference", requires="mlx-gpu")
def speed(ctx):
    ctx.check_speed("scale", ctx.randn(256, 256), 2.0, max_ratio=20.0, tokens=256)
