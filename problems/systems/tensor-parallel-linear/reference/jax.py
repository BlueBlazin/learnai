from itertools import pairwise

import jax


def shard_weights(
    w_up: jax.Array,
    b_up: jax.Array,
    w_down: jax.Array,
    b_down: jax.Array,
    n: int,
) -> list[dict[str, jax.Array]]:
    h = w_up.shape[0]
    bounds = [i * h // n for i in range(n + 1)]
    # Both layers are cut along the hidden dimension: the rows of w_up (its outputs)
    # and the columns of w_down (its inputs). The output bias is not split.
    return [
        {
            "up": w_up[lo:hi],
            "up_bias": b_up[lo:hi],
            "down": w_down[:, lo:hi],
            "down_bias": b_down,
        }
        for lo, hi in pairwise(bounds)
    ]


def tp_mlp(x: jax.Array, devs) -> jax.Array:
    parts = []
    for i in range(devs.n_devices):
        # Device i: its slice of the hidden layer, activation included, needs nothing
        # from the other devices ...
        hidden = jax.nn.silu(x @ devs.get(i, "up").T + devs.get(i, "up_bias"))
        # ... and its columns of w_down turn that slice into a partial output.
        parts.append(hidden @ devs.get(i, "down").T)
    # The only communication: one sum of tensors of the output's size. The bias is
    # added once, after it.
    return devs.all_reduce(parts) + devs.get(0, "down_bias")
