import numpy as np

from learnai.judge import case, check


@case("a 3-4-5 triangle", public=True)
def worked_example(ctx):
    ctx.check("normalize", np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32), axis=-1)


@case("every output vector has unit length and keeps its direction")
def unit_length(ctx):
    def prop(rng):
        shape = tuple(int(n) for n in rng.integers(2, 6, size=rng.integers(1, 4)))
        axis = int(rng.integers(-len(shape), len(shape)))
        x = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-2, 2)))
        y = ctx.call("normalize", x, axis=axis)
        check.shape(y, x.shape, label="normalize")
        norms = np.linalg.norm(y, axis=axis)
        check.allclose(norms, np.ones_like(norms), label="norm of the output")
        scale = np.linalg.norm(x, axis=axis, keepdims=True)
        check.allclose(y * scale, x, rtol=1e-3, atol=1e-5, label="output times input norm")

    check.forall(prop, trials=10)
